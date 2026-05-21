"""
mock_phone.py
模拟 Android App 的行为，通过 adb 和模拟器交互。
用于在开发真正的无障碍 App 之前，先跑通整个链路。
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET

import websockets

# ── 配置 ──────────────────────────────────────────────
SERVER_URL = "ws://localhost:8765"
DEVICE = "R5CR20ECSQV"                    # ← 填入 adb devices 显示的设备ID，例如 "R5CW309XXXXX"
SCREENSHOT_DIR = "./mock_screenshots"
TEMP_DIR = "/sdcard"            # 手机端临时目录
ADB_PATH = r"D:\soft\sofeware\Android\Sdk\platform-tools\adb.exe"
# ──────────────────────────────────────────────────────


def adb(cmd: str, timeout: int = 30) -> str:
    """同步 adb 调用（仅供内部使用，对外请用 adb_async）"""
    full = f"{ADB_PATH} -s {DEVICE} {cmd}"
    try:
        result = subprocess.run(full, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"[adb ERROR] {result.stderr.strip()}")
            return "ERROR"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"[adb TIMEOUT] 命令超时（{timeout}s）：{cmd}")
        return "TIMEOUT"


async def adb_async(cmd: str, timeout: int = 30) -> str:
    """在线程池中异步执行 adb，不阻塞事件循环"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, adb, cmd, timeout)


# ── 设备屏幕尺寸（启动时探测一次并缓存）────────────────────
_SCREEN_SIZE_CACHE: list[int] | None = None
_DEFAULT_SCREEN_SIZE: list[int] = [1080, 2340]   # 探测失败时的兜底值


async def get_screen_size() -> list[int]:
    """
    动态获取设备物理分辨率 [width, height]，缓存结果。
    优先解析 `wm size` 中的 'Override size'（用户改过分辨率时），其次 'Physical size'。
    任何失败都退回 _DEFAULT_SCREEN_SIZE，并打印警告。
    """
    global _SCREEN_SIZE_CACHE
    if _SCREEN_SIZE_CACHE is not None:
        return _SCREEN_SIZE_CACHE

    raw = await adb_async("shell wm size", timeout=10)
    if raw in ("ERROR", "TIMEOUT") or not raw:
        print(f"[ScreenSize] wm size 调用失败（{raw}），使用默认值 {_DEFAULT_SCREEN_SIZE}")
        _SCREEN_SIZE_CACHE = list(_DEFAULT_SCREEN_SIZE)
        return _SCREEN_SIZE_CACHE

    # wm size 可能输出:
    #   Physical size: 1080x2340
    #   Override size: 720x1560    (可选)
    # 优先 Override（实际生效），其次 Physical
    sizes: dict[str, list[int]] = {}
    for line in raw.splitlines():
        m = re.search(r"(Physical|Override)\s+size:\s*(\d+)x(\d+)", line, re.IGNORECASE)
        if m:
            label = m.group(1).lower()
            sizes[label] = [int(m.group(2)), int(m.group(3))]
    parsed = sizes.get("override") or sizes.get("physical")
    if not parsed:
        print(f"[ScreenSize] 无法解析 wm size 输出：{raw[:120]}，使用默认值")
        parsed = list(_DEFAULT_SCREEN_SIZE)
    else:
        print(f"[ScreenSize] 设备分辨率：{parsed[0]}x{parsed[1]}")
    _SCREEN_SIZE_CACHE = parsed
    return _SCREEN_SIZE_CACHE


# ── 截图 ──────────────────────────────────────────────

async def take_screenshot(step: int) -> str:
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    remote = f"{TEMP_DIR}/mock_screen.png"
    local = os.path.join(SCREENSHOT_DIR, f"step_{step:03d}.png")
    local_compressed = os.path.join(SCREENSHOT_DIR, f"step_{step:03d}_small.jpg")

    print("[截图] screencap...")
    ret = await adb_async(f"shell screencap -p {remote}", timeout=20)
    if ret == "TIMEOUT":
        raise RuntimeError("screencap 超时")

    print("[截图] pull...")
    pull_ret = await adb_async(f"pull {remote} {local}", timeout=20)
    if pull_ret in ("ERROR", "TIMEOUT") or not os.path.exists(local):
        raise RuntimeError(f"pull 截图到本地失败（adb 返回：{pull_ret}）")

    # 压缩截图（在线程池中执行 PIL 操作）
    def compress():
        from PIL import Image
        img = Image.open(local)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        max_width = 720
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        img.save(local_compressed, "JPEG", quality=75)
        with open(local_compressed, "rb") as f:
            return base64.b64encode(f.read()).decode()

    print("[截图] 压缩中...")
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, compress)
    print("[截图] 完成")
    return data


# ── UI 树解析 ──────────────────────────────────────────

async def get_ui_elements() -> list[dict]:
    remote_xml = f"{TEMP_DIR}/mock_ui.xml"
    local_xml = os.path.join(SCREENSHOT_DIR, "ui.xml")
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    await adb_async(f"shell uiautomator dump {remote_xml}", timeout=20)
    await adb_async(f"pull {remote_xml} {local_xml}", timeout=20)

    if not os.path.exists(local_xml):
        return []

    def parse_xml() -> list[dict]:
        elements = []
        index = [1]

        def _is_interactive(n) -> bool:
            """判断节点本身是否可交互（可点击 / 可编辑 / 可滚动）"""
            c = n.attrib.get("clickable") == "true"
            e = n.attrib.get("focusable") == "true" and "EditText" in n.attrib.get("class", "")
            s = n.attrib.get("scrollable") == "true"
            return c or e or s

        def _collect_child_info(node, texts: list, descs: list):
            """
            递归收集不可交互子孙节点的 text / content-desc，
            遇到独立可交互节点（将拥有自己的 index）则停止深入。
            """
            for child in node:
                if _is_interactive(child):
                    # 独立可交互子节点：有自己的 index，不合并到父节点
                    continue
                t = child.attrib.get("text", "").strip()
                d = child.attrib.get("content-desc", "").strip()
                if t and t not in texts:
                    texts.append(t)
                if d and d not in descs:
                    descs.append(d)
                _collect_child_info(child, texts, descs)

        def parse_node(node):
            bounds_str = node.attrib.get("bounds", "")
            if not bounds_str:
                for child in node:
                    parse_node(child)
                return

            try:
                parts = bounds_str.replace("][", ",").strip("[]").split(",")
                x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            except Exception:
                for child in node:
                    parse_node(child)
                return

            clickable = node.attrib.get("clickable") == "true"
            editable = node.attrib.get("focusable") == "true" and \
                       "EditText" in node.attrib.get("class", "")
            scrollable = node.attrib.get("scrollable") == "true"

            if clickable or editable or scrollable:
                # ── 收集自身 text / desc ──────────────────────────
                base_text = node.attrib.get("text", "").strip()
                base_desc = node.attrib.get("content-desc", "").strip()

                # ── 向下收集不可交互子孙的 text / desc ──────────────
                child_texts: list[str] = []
                child_descs: list[str] = []
                _collect_child_info(node, child_texts, child_descs)

                # ── 去重合并：父节点信息优先，子节点追加 ────────────
                merged_texts = []
                if base_text:
                    merged_texts.append(base_text)
                for t in child_texts:
                    if t and t not in merged_texts:
                        merged_texts.append(t)

                merged_descs = []
                if base_desc:
                    merged_descs.append(base_desc)
                for d in child_descs:
                    if d and d not in merged_descs:
                        merged_descs.append(d)

                elements.append({
                    "index": index[0],
                    "class": node.attrib.get("class", "").split(".")[-1],
                    "text": " | ".join(merged_texts),
                    "content_desc": " | ".join(merged_descs),
                    "resource_id": node.attrib.get("resource-id", ""),
                    "bounds": [x1, y1, x2, y2],
                    "clickable": clickable,
                    "editable": editable,
                    "scrollable": scrollable,
                })
                index[0] += 1

            # 无论当前节点是否可交互，都继续递归子节点
            # 让独立可交互的子节点也能获得自己的 index
            for child in node:
                parse_node(child)

        try:
            tree = ET.parse(local_xml)
            parse_node(tree.getroot())
        except Exception as e:
            print(f"[XML解析错误] {e}")

        return elements

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, parse_xml)


# ── 动作执行 ───────────────────────────────────────────

# 各类动作的等待参数：min 是固定最小等待，max 是智能等待的上限
# 智能等待会在 min 之后开始轮询，一旦检测到界面稳定就提前返回
_WAIT_PROFILES = {
    "click":      {"min": 0.25, "max": 1.5},
    "long_click": {"min": 0.30, "max": 2.0},
    "type":       {"min": 0.30, "max": 1.5},
    "scroll":     {"min": 0.30, "max": 1.5},
    "back":       {"min": 0.20, "max": 1.0},
    "home":       {"min": 0.20, "max": 1.0},
    "open_app":   {"min": 0.50, "max": 5.0},  # 应用启动差异大，上限留足
    "search_web": {"min": 0.50, "max": 4.5},
}


async def _foreground_signature() -> str:
    """
    用 dumpsys window 拿当前焦点 window 作为"界面指纹"。
    比 uiautomator dump 快 5-10 倍（~150ms vs ~800ms），适合做稳定性轮询。
    """
    result = await adb_async("shell dumpsys window windows", timeout=8)
    m = re.search(r'mCurrentFocus=Window\{[^}]*\}', result)
    return m.group(0) if m else result[:200]


async def wait_until_stable(action: str,
                            expect_pkg: str | None = None,
                            min_wait: float | None = None,
                            max_wait: float | None = None) -> None:
    """
    动作执行后智能等待界面稳定：
    - 始终先固定等 min_wait（让点击/输入事件冒泡完成）
    - 之后每 250ms 取一次"焦点 window 指纹"，连续两次相同 → 视为稳定，提前返回
    - 若指定 expect_pkg，则在到达 max_wait 之前如果前台包不等于目标，会继续等待
    - 最长不超过 max_wait

    通过这套机制，open_app 类长 sleep 通常能从 3.5s 缩短到 1-2s。
    """
    profile = _WAIT_PROFILES.get(action, {"min": 0.3, "max": 1.5})
    min_w = profile["min"] if min_wait is None else min_wait
    max_w = profile["max"] if max_wait is None else max_wait

    await asyncio.sleep(min_w)
    elapsed = min_w
    last_sig: str | None = None
    poll = 0.25

    while elapsed < max_w:
        sig = await _foreground_signature()
        # 包名期望未满足，强制继续轮询
        pkg_ok = True
        if expect_pkg:
            pkg_ok = expect_pkg in sig
        if last_sig is not None and sig == last_sig and pkg_ok:
            return  # 稳定
        last_sig = sig
        await asyncio.sleep(poll)
        elapsed += poll
    # 超时也直接返回，让上层逻辑用下一次 get_ui_elements 兜底


async def do_click(x: int, y: int):
    print(f"[执行] click ({x}, {y})")
    await adb_async(f"shell input tap {x} {y}")
    await wait_until_stable("click")


async def do_long_click(x: int, y: int):
    print(f"[执行] long_click ({x}, {y})")
    await adb_async(f"shell input swipe {x} {y} {x} {y} 1000")
    await wait_until_stable("long_click")


async def do_type(x: int, y: int, text: str):
    print(f"[执行] type ({x}, {y}) -> '{text}'")
    await adb_async(f"shell input tap {x} {y}")
    await asyncio.sleep(0.3)

    if _is_ascii(text):
        # 纯 ASCII：直接用 adb input text
        # 空格用 %s 转义；单引号用 shell 转义序列 '\'' 处理，避免直接删除
        safe_text = text.replace(" ", "%s").replace("'", "'\\''")
        await adb_async(f"shell input text '{safe_text}'")
    else:
        # 含中文/特殊字符：优先尝试 ADBKeyboard broadcast
        result = await _type_with_adbkeyboard(text)
        if not result:
            # 降级：逐字符用 keyevent，仅适用于 ASCII 部分
            print("[警告] ADBKeyboard 不可用，尝试降级输入（中文可能丢失）")
            safe_text = "".join(c if ord(c) < 128 else "" for c in text)
            safe_text = safe_text.replace(" ", "%s").replace("'", "'\\''")
            if safe_text:
                await adb_async(f"shell input text '{safe_text}'")

    await wait_until_stable("type")


def _is_ascii(text: str) -> bool:
    return all(ord(c) < 128 for c in text)


async def _type_with_adbkeyboard(text: str) -> bool:
    """
    通过 ADBKeyboard 广播输入中文/特殊字符。
    需要在设备上安装 ADBKeyboard 并设置为默认输入法：
      adb install ADBKeyboard.apk
      adb shell ime set com.android.adbkeyboard/.AdbIME
    返回 True 表示成功，False 表示 ADBKeyboard 不可用。
    """
    safe = text.replace("'", "\\'")
    result = await adb_async(
        f"shell am broadcast -a ADB_INPUT_TEXT --es msg '{safe}'"
    )
    # ADBKeyboard 响应正常时返回 "result=0" 或包含 "Broadcast completed"
    if "result=0" in result or "Broadcast completed" in result:
        return True
    print(f"[ADBKeyboard] 响应异常：{result}")
    return False


async def do_scroll(x: int, y: int, direction: str):
    print(f"[执行] scroll ({x}, {y}) -> {direction}")
    dist = 400
    offsets = {
        "up":    (0, -dist),
        "down":  (0, dist),
        "left":  (-dist, 0),
        "right": (dist, 0),
    }
    dx, dy = offsets.get(direction, (0, -dist))
    await adb_async(f"shell input swipe {x} {y} {x+dx} {y+dy} 300")
    await wait_until_stable("scroll")


async def do_back():
    print("[执行] back")
    await adb_async("shell input keyevent KEYCODE_BACK")
    await wait_until_stable("back")


async def do_home():
    print("[执行] home")
    await adb_async("shell input keyevent KEYCODE_HOME")
    await wait_until_stable("home")


async def get_foreground_package() -> str:
    """获取当前前台应用的包名"""
    result = await adb_async("shell dumpsys window windows", timeout=10)
    m = re.search(r'mCurrentFocus=Window\{[^\s]+ [^\s]+ ([^/}\s]+)', result)
    return m.group(1) if m else ""


async def _resolve_launcher_component(package: str) -> str | None:
    """
    通过 cmd package resolve-activity 拿到目标应用的 LAUNCHER 组件名。
    返回形如 "com.foo/.MainActivity"，失败时返回 None。
    """
    out = await adb_async(
        f"shell cmd package resolve-activity --brief -c android.intent.category.LAUNCHER {package}",
        timeout=10,
    )
    if out in ("ERROR", "TIMEOUT") or not out:
        return None
    # 输出最后一行通常是 "com.foo/.bar.MainActivity"
    for line in reversed(out.splitlines()):
        line = line.strip()
        if "/" in line and " " not in line and line.startswith(package):
            return line
    return None


async def do_open_app(package: str):
    """
    优先用 `am start -n <component>` 启动（不会被反 monkey 检测拦截，
    部分银行/支付/IM 应用对 monkey 启动会拒绝）。
    解析 LAUNCHER 组件失败时回退 monkey。
    """
    print(f"[执行] open_app {package}")
    component = await _resolve_launcher_component(package)
    started = False
    if component:
        result = await adb_async(f"shell am start -n {component}", timeout=10)
        # am start 成功输出包含 "Starting: Intent"；失败时多带 "Error" 或 "Exception"
        if result not in ("ERROR", "TIMEOUT") and "Error" not in result and "Exception" not in result:
            started = True
            print(f"[open_app] am start 启动：{component}")
        else:
            print(f"[open_app] am start 失败（{result[:80]}），降级为 monkey")
    if not started:
        await adb_async(f"shell monkey -p {package} -c android.intent.category.LAUNCHER 1")
    # 智能等待：一旦前台包变为目标且界面稳定即返回，最长 5s
    await wait_until_stable("open_app", expect_pkg=package)


async def do_search_web(query: str):
    """直接用 Chrome 打开 Google 搜索，绕过地址栏交互"""
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"https://www.google.com/search?q={encoded}"
    print(f"[执行] search_web: {url}")
    await adb_async(f'shell am start -a android.intent.action.VIEW -d "{url}" com.android.chrome')
    await wait_until_stable("search_web", expect_pkg="com.android.chrome")


# ── WebSocket 主循环 ────────────────────────────────────

async def handle_session(ws, step_counter: list):
    """处理单次连接的消息循环，遇到 finish 返回，断开连接则抛出异常"""
    print("[MockPhone] 已连接，等待服务端分配任务...")

    async for raw in ws:
        msg = json.loads(raw)
        msg_type = msg.get("type")

        # 收到任务
        if msg_type == "task":
            print(f"\n[任务] {msg.get('task')}")

        # 请求当前界面状态
        elif msg_type == "request_state":
            print("[状态] 获取 UI 树...")
            elements, foreground_pkg, screen_size = await asyncio.gather(
                get_ui_elements(),
                get_foreground_package(),
                get_screen_size(),
            )
            print(f"[状态] 共 {len(elements)} 个可交互元素，前台应用：{foreground_pkg}")
            await ws.send(json.dumps({
                "type": "state",
                "package": foreground_pkg,
                "screen_size": screen_size,
                "ui_elements": elements,
            }))

        # 请求截图
        elif msg_type == "request_screenshot":
            print("[截图] 正在截图...")
            b64 = await take_screenshot(step_counter[0])
            step_counter[0] += 1
            await ws.send(json.dumps({
                "type": "screenshot",
                "screenshot": b64,
            }))

        # 执行点击
        elif msg_type == "click":
            await do_click(msg["x"], msg["y"])
            await ws.send(json.dumps({"type": "result", "status": "success"}))

        # 执行长按
        elif msg_type == "long_click":
            await do_long_click(msg["x"], msg["y"])
            await ws.send(json.dumps({"type": "result", "status": "success"}))

        # 执行输入
        elif msg_type == "type":
            await do_type(msg["x"], msg["y"], msg.get("text", ""))
            await ws.send(json.dumps({"type": "result", "status": "success"}))

        # 执行滚动
        elif msg_type == "scroll":
            await do_scroll(msg["x"], msg["y"], msg.get("direction", "up"))
            await ws.send(json.dumps({"type": "result", "status": "success"}))

        # 返回键
        elif msg_type == "back":
            await do_back()
            await ws.send(json.dumps({"type": "result", "status": "success"}))

        # 主屏键
        elif msg_type == "home":
            await do_home()
            await ws.send(json.dumps({"type": "result", "status": "success"}))

        # 打开应用
        elif msg_type == "open_app":
            await do_open_app(msg.get("package", ""))
            await ws.send(json.dumps({"type": "result", "status": "success"}))

        # 直接搜索网页
        elif msg_type == "search_web":
            await do_search_web(msg.get("query", ""))
            await ws.send(json.dumps({"type": "result", "status": "success"}))

        # 查询设备上安装的包名
        elif msg_type == "find_package":
            keyword = msg.get("keyword", "").lower()
            all_pkgs = await adb_async("shell pm list packages", timeout=15)
            packages = [
                line.replace("package:", "").strip()
                for line in all_pkgs.splitlines()
                if line.startswith("package:") and keyword in line.lower()
            ]
            print(f"[执行] find_package '{keyword}' → {packages}")
            await ws.send(json.dumps({"type": "package_result", "packages": packages}))

        # 任务完成 —— 断开后重连，等待下一个任务
        elif msg_type == "finish":
            print(f"[完成] {msg.get('message', '任务完成')}")
            print("[MockPhone] 断开连接，准备重连等待下一个任务...\n")
            return

        else:
            print(f"[未知指令] {msg_type}")


async def run():
    step_counter = [0]   # 用列表包装，方便跨函数共享计数
    retry_delay = 2      # 重连等待秒数

    while True:
        try:
            print(f"[MockPhone] 连接到 {SERVER_URL} ...")
            async with websockets.connect(SERVER_URL) as ws:
                await handle_session(ws, step_counter)
        except websockets.ConnectionClosed:
            print(f"[MockPhone] 连接已关闭，{retry_delay}s 后重连...")
        except OSError as e:
            print(f"[MockPhone] 无法连接服务器（{e}），{retry_delay}s 后重试...")
        except Exception as e:
            print(f"[MockPhone] 异常：{e}，{retry_delay}s 后重连...")

        await asyncio.sleep(retry_delay)


if __name__ == "__main__":
    asyncio.run(run())
