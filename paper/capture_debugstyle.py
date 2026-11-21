"""Capture demo screenshots - uses the exact transport pattern from the working
diagnostic probes (sequential ids, timeout=60, short evaluates, python-side polling).

Shots (all real interactions on localhost:8501):
  1  main interface (Hausa default)
  2  Hausa true-positive pair -> paraphrase verdict
  3  language switched to Yoruba, true-negative pair -> not-paraphrase verdict
Output: paper/presentation_assets/screenshot_{1,2,3}.png
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
from pathlib import Path

import requests
import websocket

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "presentation_assets"
ASSETS.mkdir(exist_ok=True)
CASE = json.loads((HERE / "case_study.json").read_text(encoding="utf-8"))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9341

ws = None
_n = [0]


def ev(expr, ap=True):
    """Evaluate; exact transport of the working debug probes."""
    _n[0] += 1
    ws.send(json.dumps({"id": _n[0], "method": "Runtime.evaluate",
                        "params": {"expression": "(async () => {\n" + expr + "\n})()",
                                   "awaitPromise": ap, "returnByValue": True}}))
    while True:
        m = json.loads(ws.recv())
        if m.get("id") == _n[0]:
            if "error" in m:
                return f"ERR:{m['error']['message'][:100]}"
            if m["result"].get("exceptionDetails"):
                return "EXC:" + str(m["result"]["exceptionDetails"])[:150]
            res = m["result"]
            return res.get("result", {}).get("value") if isinstance(res.get("result"), dict) else res.get("data")


def shot(path: Path):
    _n[0] += 1
    ws.send(json.dumps({"id": _n[0], "method": "Page.captureScreenshot", "params": {"format": "png"}}))
    while True:
        m = json.loads(ws.recv())
        if m.get("id") == _n[0]:
            data = m.get("result", {}).get("data")
            assert data, f"screenshot null: {str(m)[:200]}"
            path.write_bytes(base64.b64decode(data))
            print(f"wrote {path.name}")
            return


def fill_verified(cdp_s1, cdp_s2, s1, s2):
    """Fill both text areas, then verify they survive Streamlit reruns; refill if not."""
    for attempt in range(4):
        print("fill+click:", ev(SET_VALS % (json.dumps(s1), json.dumps(s2))))
        time.sleep(2.0)
        ok = ev(
            "return (document.querySelectorAll('textarea')[0]?.value.slice(0, 25) === "
            + json.dumps(s1[:25]) + " && document.querySelectorAll('textarea')[1]?.value.slice(0, 25) === "
            + json.dumps(s2[:25]) + ") ? 'match' : 'mismatch';")
        print(f"  verify[{attempt}]:", ok)
        if ok == "match":
            return True
    return False


def wait_render(marker, tries):
    for _ in range(tries):
        if ev(f"return document.body ? document.body.innerText.includes({json.dumps(marker)}) : false;") is True:
            return True
        time.sleep(5)
    return False


PRESS = """
function press(el) {
  for (const t of ["pointerover", "pointerenter", "pointerdown", "mousedown", "pointerup", "mouseup", "click"]) {
    const E = t.startsWith("pointer") ? window.PointerEvent : MouseEvent;
    el.dispatchEvent(new E(t, {bubbles: true, cancelable: true, view: window}));
  }
}
function main() {
  const sel = document.querySelector('[data-testid="stSelectbox"] [role="combobox"]')
              || document.querySelectorAll('[role="combobox"]')[0];
  if (!sel) return 'no select';
  press(sel);
  return 'opened';
}
return main();
"""

PRESS_PICKER = """
function press(el) {
  for (const t of ["pointerover", "pointerenter", "pointerdown", "mousedown", "pointerup", "mouseup", "click"]) {
    const E = t.startsWith("pointer") ? window.PointerEvent : MouseEvent;
    el.dispatchEvent(new E(t, {bubbles: true, cancelable: true, view: window}));
  }
}
function main() {
  const boxes = document.querySelectorAll('[role="combobox"]');
  const sel = boxes[boxes.length - 1];   // example picker is the last combobox
  if (!sel) return 'no picker';
  press(sel);
  return 'opened';
}
return main();
"""

PRESS_OPTION_EX2 = """
function press(el) {
  for (const t of ["pointerover", "pointerenter", "pointerdown", "mousedown", "pointerup", "mouseup", "click"]) {
    const E = t.startsWith("pointer") ? window.PointerEvent : MouseEvent;
    el.dispatchEvent(new E(t, {bubbles: true, cancelable: true, view: window}));
  }
}
function main() {
  const opts = [...document.querySelectorAll('[role="option"]')]
    .filter(li => li.textContent.trim() === 'Example 2');
  if (!opts.length) return 'no option';
  press(opts[0]);
  return 'picked';
}
return main();
"""

PRESS_OPTION = """
function press(el) {
  for (const t of ["pointerover", "pointerenter", "pointerdown", "mousedown", "pointerup", "mouseup", "click"]) {
    const E = t.startsWith("pointer") ? window.PointerEvent : MouseEvent;
    el.dispatchEvent(new E(t, {bubbles: true, cancelable: true, view: window}));
  }
}
function main() {
  const opts = [...document.querySelectorAll('[role="option"]')]
    .filter(li => li.textContent.trim() === 'Yoruba');
  if (!opts.length) return 'no option';
  press(opts[0]);
  return 'switched';
}
return main();
"""

FILL_VALS = """
async function main(s1, s2) {
  function setTA(el, val) {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(el, val);
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
    el.blur();
    el.dispatchEvent(new FocusEvent('blur', {bubbles: true}));
  }
  const tas = document.querySelectorAll('textarea');
  if (tas.length < 2) return 'no textareas';
  setTA(tas[0], s1); setTA(tas[1], s2);
  return 'filled';
}
return await main(%s, %s);
"""

CLICK = """
function main() {
  const btn = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Compare sentences');
  if (!btn) return 'no button';
  btn.click();
  return 'clicked';
}
return main();
"""


def find_case(lang, kind):
    return next(c for c in CASE[lang]["cases"] if c["kind"] == kind)


def main():
    global ws
    profile = "/tmp/chrome-cdp-ap3"
    shutil.rmtree(profile, ignore_errors=True)
    proc = subprocess.Popen([
        CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
        "--remote-allow-origins=*", f"--user-data-dir={profile}", "--no-first-run",
        "--window-size=1440,900", "http://localhost:8501",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(40):
            time.sleep(0.5)
            try:
                t = requests.get(f"http://localhost:{PORT}/json/list", timeout=2).json()
                pages = [x for x in t if x.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
        assert ws_url, "no CDP target"
        ws = websocket.create_connection(ws_url, timeout=60)
        time.sleep(14)
        assert wait_render("Compare sentences", 24), "app did not render"
        time.sleep(1.5)
        shot(ASSETS / "screenshot_1_main.png")

        hau = find_case("hau", "true_positive")
        print("fill:", ev(FILL_VALS % (json.dumps(hau["sentence1"]), json.dumps(hau["sentence2"]))))
        time.sleep(2.5)
        print("click:", ev(CLICK))
        assert wait_render("calibrated similarity score", 48), "prediction did not render"
        got = ev("return (document.body.innerText.match(/calibrated probability ([0-9.]+)/) || [])[1] || 'none';")
        print("verdict prob:", got, "| expected:", hau["calibrated_prob"])
        assert abs(float(got) - hau["calibrated_prob"]) < 0.02, "verdict does not match offline predictor"
        time.sleep(1.2)
        shot(ASSETS / "screenshot_2_hausa_paraphrase.png")

        print("select:", ev(PRESS))
        time.sleep(1.5)
        print("switch:", ev(PRESS_OPTION))
        time.sleep(4)
        print("panel:", ev("return document.body.innerText.includes('constructed pairs') ? 'yoruba' : 'still-hausa';"))
        # shot 3 via the app's OWN example picker (native widget -> server state
        # updates reliably; DOM-synthesised textarea input races the server).
        import math
        demo = json.loads((HERE.parent / "src/afripara/app/demo_examples.json").read_text())
        ex2 = demo["Yoruba"]["pairs"][1]  # gold 0 (not paraphrase)
        exp_cal = 1 / (1 + math.exp(-math.log(ex2["score"] / (1 - ex2["score"])) / CASE["yor"]["temperature"]))
        print("picker open:", ev(PRESS_PICKER))
        time.sleep(1.2)
        print("pick ex2:", ev(PRESS_OPTION_EX2))
        time.sleep(2.5)
        print("click:", ev(CLICK))
        assert wait_render("calibrated similarity score", 48), "yoruba prediction did not render"
        got = ev("return (document.body.innerText.match(/calibrated probability ([0-9.]+)/) || [])[1] || 'none';")
        print("verdict prob:", got, "| expected:", round(exp_cal, 3))
        assert abs(float(got) - exp_cal) < 0.02, "verdict does not match the recorded example score"
        time.sleep(1.2)
        shot(ASSETS / "screenshot_3_yoruba_not_paraphrase.png")
        ws.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
