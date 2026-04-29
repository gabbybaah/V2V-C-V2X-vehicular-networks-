# =============================================================================
# handshake_v6/dashboard_road.py  —  Road Dashboard (V6 Redesign)
#
# What's new:
#   - Emoji car icons for every vehicle type
#   - Live Activity panel: every car listed with emoji + plain-English status
#   - Compact controls bar always visible at bottom
#   - School zone and roadworks labels so clusters make sense
#   - Near-miss / collision flash in header
#   - Remote (multi-VM) cars clearly labelled with arrow
# =============================================================================
import time
from rich.console import Console
from rich.live    import Live
from rich.table   import Table
from rich.panel   import Panel
from rich.text    import Text
from rich.layout  import Layout
from rich         import box
from road_geometry import RG, ROAD_LENGTH_M
from config import CarType, RState, WeatherState, WEATHER_PARAMS

console = Console()

# ── Emoji map ─────────────────────────────────────────────────────────────────
# Cars are shown as right-facing arrows — colour distinguishes type.
# P1 = bright yellow ★, P2 = bright blue ★, Smart = yellow ►,
# Legacy = white ►, Emergency = 🚨, Rogue = magenta ►, Platoon = magenta ►
_ARROW  = "►"   # NPC arrow
_STAR   = "★"   # Player star — unmissable

_EMOJI = {
    CarType.SMART:     "►",
    CarType.LEGACY:    "►",
    CarType.EMERGENCY: "►",
    CarType.ROGUE:     "►",
    CarType.BREAKDOWN: "►",
    CarType.PLATOON:   "►",
    "PLAYER1":         "★",   # P1 yellow star
    "PLAYER2":         "★",   # P2 blue star
}

# Colours per car type
_TYPE_COLORS = {
    CarType.SMART:     "bright_yellow",
    CarType.LEGACY:    "white",
    CarType.EMERGENCY: "bold bright_red",
    CarType.ROGUE:     "bold bright_magenta",
    CarType.BREAKDOWN: "dim red",
    CarType.PLATOON:   "bright_magenta",
}

# Player colours — stars, very distinct
_PLAYER_COLORS = {
    1: "bold yellow",          # P1 — gold/yellow star ★
    2: "bold bright_blue",     # P2 — blue star ★
}

def _player_machine_id(car):
    """Extract machine_id from car dict — try field first, then parse label."""
    mid = car.get("machine_id", 0)
    if mid: return mid
    label = car.get("label", "")
    if label.startswith("P") and len(label) > 1 and label[1].isdigit():
        return int(label[1])
    return 1

def _car_emoji(car):
    """Return the glyph for a car: 🚨 for emergency, ★ for player, ► otherwise."""
    if car.get("is_player"):
        return _STAR
    if car.get("car_type") == CarType.EMERGENCY:
        return "🚨"
    return _ARROW

def _car_style(car):
    """Return Rich style string for a car — colour is how you tell them apart."""
    if car.get("is_player"):
        mid = _player_machine_id(car)
        return _PLAYER_COLORS.get(mid, "bold bright_green")
    ct = car.get("car_type", CarType.SMART)
    return _TYPE_COLORS.get(ct, car.get("profile_color", "white"))


# ── Plain-English state ───────────────────────────────────────────────────────
_STATE_PLAIN = {
    RState.DRIVING:     "driving",
    RState.OVERTAKING:  "overtaking",
    RState.LANE_CHANGE: "changing lane",
    RState.BRAKING:     "braking hard",
    RState.YIELDING:    "yielding to emergency",
    RState.BROKEN_DOWN: "broken down ⚠",
    RState.PLATOONING:  "platooning",
    RState.SCHOOL_ZONE: "school zone — slowing",
    RState.SHOULDER:    "on shoulder (clearing emergency)",
    RState.DONE:        "finished",
}

def _state_plain(car):
    s = car.get("state","")
    txt = _STATE_PLAIN.get(s, s.lower().replace("_"," "))
    if s == RState.LANE_CHANGE:
        tgt = car.get("target_lane", car.get("lane",0))
        lname = ["slow","fast","shoulder"][tgt] if tgt < 3 else str(tgt)
        txt = f"moving to {lname} lane"
    return txt

def _lane_name(n):
    return ["Slow lane","Fast lane","Shoulder"][n] if n < 3 else f"Lane {n}"

def _fuel_str(fuel):
    if fuel <= 0.0:  return "[bold red]EMPTY[/bold red]"
    if fuel <= 0.10: return f"[bold yellow]{fuel*100:.0f}%⚠[/bold yellow]"
    if fuel <= 0.25: return f"[yellow]{fuel*100:.0f}%[/yellow]"
    return f"[dim]{fuel*100:.0f}%[/dim]"

# ── Weather styles ────────────────────────────────────────────────────────────
_WEATHER_BG = {
    WeatherState.CLEAR:      "grey11",
    WeatherState.LIGHT_RAIN: "grey15",
    WeatherState.HEAVY_RAIN: "grey19",
    WeatherState.FOG:        "grey23",
    WeatherState.LIGHT_SNOW: "grey15",
    WeatherState.ICE:        "dark_red",
}
_WEATHER_FG = {
    WeatherState.CLEAR:      "bold white",
    WeatherState.LIGHT_RAIN: "bold white",
    WeatherState.HEAVY_RAIN: "bold bright_white",
    WeatherState.FOG:        "bold bright_yellow",
    WeatherState.LIGHT_SNOW: "bold cyan",
    WeatherState.ICE:        "bold bright_red",
}


# ═══════════════════════════════════════════════════════════════════════════════
# ROAD STRIP  (emoji double-width per car)
# ═══════════════════════════════════════════════════════════════════════════════

_PRI = {
    "PLAYER":         0,
    CarType.EMERGENCY:1,
    CarType.PLATOON:  2,
    CarType.SMART:    3,
    CarType.BREAKDOWN:4,
    CarType.LEGACY:   5,
}

def _build_road_strip(cars_in_lane, strip_cols, road_len=ROAD_LENGTH_M,
                      transitioning_cars=None, lane_n=0):
    """
    Arrow-based road strip. Each slot = 2 terminal columns: ► + space.
    Colour is how you tell cars apart — P1=bright green, P2=cyan, emerg=red, etc.
    transitioning_cars: list of cars doing a lane change that cross this lane.
    """
    n_pos = max(20, strip_cols // 2)
    slots = {}

    for car in cars_in_lane:
        pos = car.get("road_pos_m", 0.0)
        ct  = car.get("car_type", CarType.SMART)
        key = "PLAYER" if car.get("is_player") else ct
        pri = _PRI.get(key, 6)
        idx = max(0, min(n_pos-1, int(pos / road_len * n_pos)))
        if idx not in slots or pri < slots[idx][2]:
            slots[idx] = (_car_style(car), pri, False, key)

    # Cars mid-lane-change that are between this lane and another (ghost)
    if transitioning_cars:
        for car in transitioning_cars:
            pos   = car.get("road_pos_m", 0.0)
            prog  = car.get("lc_progress", 1.0)
            if prog >= 1.0: continue
            idx = max(0, min(n_pos-1, int(pos / road_len * n_pos)))
            ct  = car.get("car_type", CarType.SMART)
            if idx not in slots:
                slots[idx] = ("dim " + _car_style(car), 7, True, ct)

    line = Text()
    for i in range(n_pos):
        if i in slots:
            clr, _, ghost, key = slots[i]
            if key == "PLAYER":
                glyph = _STAR
            elif key == CarType.EMERGENCY:
                glyph = "🚨"
            else:
                glyph = _ARROW
            if glyph == "🚨":
                # emoji is 2 terminal cols — no trailing space
                line.append(glyph, style=clr)
            else:
                line.append(glyph, style=clr)
                line.append(" ", style="default")
        else:
            line.append("· ", style="dim black")
    return line



def _build_zone_bar(strip_cols, road_len=ROAD_LENGTH_M):
    """Zone highlight bar matching the road strip width (2 chars/slot)."""
    n_pos = max(20, strip_cols // 2)
    _ZC = {
        "SCHOOL_1":  ("on yellow", "🏫"),
        "ROADWORKS": ("on red",    "🚧"),
    }
    slots = {}
    for zname, zdata in RG.ZONES.items():
        if zname not in _ZC: continue
        s = int(zdata[0] / road_len * n_pos)
        e = min(n_pos-1, int(zdata[1] / road_len * n_pos))
        style, icon = _ZC[zname]
        for col in range(s, e+1):
            slots[col] = (icon + " " if col == s else "  ", style)

    line = Text()
    for i in range(n_pos):
        if i in slots:
            ch, st = slots[i]
            line.append(ch, style=st)
        else:
            line.append("  ", style="dim")
    return line


def _build_heatmap(heatmap_data, strip_cols):
    """Speed heatmap bar: green=fast, yellow=medium, red=slow. 2 chars per slot."""
    n_pos = max(20, strip_cols // 2)
    line  = Text()
    if not heatmap_data:
        for _ in range(n_pos): line.append("  ", style="dim")
        return line
    n = len(heatmap_data)
    for i in range(n_pos):
        idx = max(0, min(n-1, int(i / n_pos * n)))
        spd = heatmap_data[idx]
        if spd <= 0:   style = "dim black"
        elif spd < 20: style = "bold red"
        elif spd < 40: style = "bold yellow"
        else:          style = "bold green"
        line.append("▌█", style=style)
    return line


def _build_ruler(strip_cols, road_len=ROAD_LENGTH_M):
    n_pos = max(20, strip_cols // 2)
    chars = ["  "] * n_pos
    for km in range(0, int(road_len / 1000) + 1):
        idx = min(int(km * 1000 / road_len * n_pos), n_pos - 1)
        chars[idx] = f"{km}k"
    return "".join(chars)



def _build_weather_bar(status):
    ws    = status.get("weather_state", WeatherState.CLEAR)
    wp    = WEATHER_PARAMS.get(ws, WEATHER_PARAMS[0])
    age   = time.time() - status.get("weather_changed_at", 0)
    flash = age < 4.0 and status.get("weather_changed_at", 0) > 0

    bg = "bright_yellow" if flash else _WEATHER_BG.get(ws, "grey11")
    fg = "bold black"    if flash else _WEATHER_FG.get(ws, "bold white")

    icon = wp["icon"]; name = wp["name"]; desc = wp["desc"]
    content = f"  {icon} {name}  —  {desc}"
    if flash: content = "  ⚠ WEATHER CHANGE ⚠  " + content

    nmiss = status.get("near_miss_count", 0)
    cols  = status.get("collision_count", 0)
    alerts = ""
    if nmiss: alerts += f"   ⚠ {nmiss} NEAR-MISS"
    if cols:  alerts += f"   💥 {cols} COLLISION"

    txt = Text()
    txt.append(content, style=f"{fg} on {bg}")
    if alerts:
        txt.append(alerts, style="bold red")
    return txt


# ═══════════════════════════════════════════════════════════════════════════════
# ROAD VISUAL PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def _build_road_panel(status, strip_cols):
    all_cars = status.get("all_cars_sorted", [])
    elapsed  = status.get("elapsed_s", 0)
    total    = status.get("total_cars", 0)
    active   = status.get("active_cars", 0)
    done_n   = status.get("done_cars", 0)
    remote   = status.get("remote_count", 0)
    rush     = "[bold yellow][RUSH HOUR][/bold yellow]" if status.get("rush_hour") else ""
    rw       = "[bold red][ROADWORKS][/bold red]"       if status.get("roadworks") else ""
    split    = "[bold magenta][SPLIT BRAIN][/bold magenta]" if status.get("split_brain") else ""
    flags    = " ".join(f for f in [rush, rw, split] if f)

    def lane(n):
        return [c for c in all_cars
                if c.get("lane") == n and c.get("state") != RState.DONE]

    # Cars currently mid-lane-change — show ghost in from-lane too
    def transitioning_from(n):
        return [c for c in all_cars
                if c.get("lc_from_lane") == n
                and c.get("lane") != n
                and c.get("lc_progress", 1.0) < 1.0
                and c.get("state") != RState.DONE]

    road_text = Text()

    road_text.append(" 🌤  Weather  │ ", style="dim")
    road_text.append_text(_build_weather_bar(status))
    road_text.append("\n")

    road_text.append(" 🏎  Fast lane│ ", style="bold cyan")
    road_text.append_text(_build_road_strip(lane(1), strip_cols,
                          transitioning_cars=transitioning_from(1), lane_n=1))
    road_text.append("\n")

    road_text.append(" 🚗  Slow lane│ ", style="bold white")
    road_text.append_text(_build_road_strip(lane(0), strip_cols,
                          transitioning_cars=transitioning_from(0), lane_n=0))
    road_text.append("\n")

    road_text.append(" 🛑  Shoulder │ ", style="dim yellow")
    road_text.append_text(_build_road_strip(lane(2), strip_cols,
                          transitioning_cars=transitioning_from(2), lane_n=2))
    road_text.append("\n")

    road_text.append(" 🏫🚧 Zones   │ ", style="dim")
    road_text.append_text(_build_zone_bar(strip_cols))
    road_text.append("  (🏫school  🚧roadworks  🔶breakdown)\n", style="dim")

    road_text.append(" 📊  Speed    │ ", style="dim green")
    road_text.append_text(_build_heatmap(status.get("heatmap", []), strip_cols))
    road_text.append("  🟢fast 🟡med 🔴slow\n", style="dim")

    road_text.append(" 📍  km ──►   │ ", style="dim cyan")
    road_text.append(_build_ruler(strip_cols), style="dim cyan")

    title = Text()
    title.append("HANDSHAKE V6 · ROAD  ", style="bold blue")
    title.append(f"T+{elapsed:.0f}s  ", style="white")
    title.append(f"🚗 {active} on road  ✓ {done_n}/{total} done  ", style="cyan")
    if remote: title.append(f"🔗 {remote} from other VM  ", style="bold cyan")
    if flags:  title.append(flags, style="bold yellow")

    return Panel(road_text, title=title, border_style="blue", padding=(0,0))


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE ACTIVITY PANEL  (every car, emoji + plain-English status)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_activity_panel(status):
    all_cars = status.get("all_cars_sorted", [])
    active   = [c for c in all_cars if c.get("state") != RState.DONE]

    tbl = Table(box=box.SIMPLE, show_header=True,
                header_style="bold dim", padding=(0,1), expand=True)
    tbl.add_column("",       width=3)    # emoji
    tbl.add_column("Car",    width=7)
    tbl.add_column("km/h",   width=5,  justify="right")
    tbl.add_column("Pos",    width=5,  justify="right")
    tbl.add_column("What is this car doing?", ratio=1)
    tbl.add_column("⛽",    width=5)

    for car in active[:26]:
        emoji  = _car_emoji(car)
        label  = car.get("label","?")[:7]
        spd    = f"{car.get('speed_kmh',0):.0f}"
        pos    = f"{car.get('road_pos_m',0)/1000:.1f}k"
        state  = _state_plain(car)
        fuel   = car.get("fuel_pct", 1.0)
        fstr   = _fuel_str(fuel)
        remote = car.get("is_remote", False)
        player = car.get("is_player", False)
        style  = _car_style(car)

        if player:
            mid = _player_machine_id(car)
            label = label  # keep original P1◆ / P2◆ label
            state = f"YOU (P{mid}) — " + state
        elif remote:
            label = label + "↗"
            style = "dim cyan"
            state = "(other VM) " + state

        tbl.add_row(emoji, label, spd, pos, state, fstr, style=style)

    if not active:
        tbl.add_row("🏁","","","","[bold green]All cars finished![/bold green]","")

    return Panel(
        tbl,
        title="[bold white]🚦 Live Traffic[/bold white]",
        subtitle="[dim][bold yellow]★[/bold yellow]=P1(you)  [bold bright_blue]★[/bold bright_blue]=P2(other)  🚨=emerg  [bright_yellow]►[/bright_yellow]=smart  [white]►[/white]=legacy  ↗=other VM[/dim]",
        border_style="white",
        padding=(0,0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CAR PROFILE PANEL  — per-car state card, plain English, one line each
# ═══════════════════════════════════════════════════════════════════════════════

# Rich descriptions for each state
_PROFILE_STATE = {
    RState.DRIVING:     ("·",  "dim",          "driving normally"),
    RState.OVERTAKING:  ("↑",  "bold green",   "overtaking"),
    RState.LANE_CHANGE: ("→",  "cyan",         "changing lane"),
    RState.BRAKING:     ("⚠",  "bold yellow",  "braking hard"),
    RState.YIELDING:    ("←",  "magenta",      "yielding to emergency"),
    RState.BROKEN_DOWN: ("⛔", "bold red",     "BROKEN DOWN"),
    RState.PLATOONING:  ("▶",  "green",        "platooning"),
    RState.SCHOOL_ZONE: ("🏫", "bright_yellow","school zone — slowing"),
    RState.SHOULDER:    ("→",  "yellow",       "on shoulder — clearing emergency"),
    RState.DONE:        ("✓",  "dim green",    "reached destination"),
}

def _build_car_profile_panel(status):
    """
    Compact per-car state card. One line per active car showing:
    emoji  label  lane  speed  ► current state in plain English
    Additive panel — nothing removed.
    """
    all_cars = status.get("all_cars_sorted", [])
    active   = [c for c in all_cars if c.get("state") != RState.DONE]

    lines = Text()
    for car in active:
        emoji  = _car_emoji(car)
        label  = car.get("label","?")
        state  = car.get("state","")
        lane   = car.get("lane",0)
        spd    = car.get("speed_kmh",0)
        pos    = car.get("road_pos_m",0)/1000
        lc_prog= car.get("lc_progress",1.0)
        player = car.get("is_player", False)
        remote = car.get("is_remote", False)
        style_c= _car_style(car)

        icon, style, desc = _PROFILE_STATE.get(state,
            ("·", "dim", state.lower().replace("_"," ")))

        # Enrich description
        if state == RState.LANE_CHANGE:
            tgt = car.get("target_lane", lane)
            lnames = ["slow","fast","shoulder"]
            tname  = lnames[tgt] if tgt < 3 else str(tgt)
            prog_pct = int(lc_prog * 100)
            desc   = f"changing lane → {tname} ({prog_pct}%)"
        elif state == RState.OVERTAKING:
            desc   = f"overtaking at {spd:.0f} km/h"
        elif state == RState.BROKEN_DOWN:
            desc   = f"BROKEN DOWN at {pos:.1f}km ⚠ hazard active"
        elif state == RState.SHOULDER:
            desc   = f"on shoulder at {pos:.1f}km — waiting for emergency to pass"
        elif state == RState.SCHOOL_ZONE:
            desc   = f"school zone — {spd:.0f} km/h"
        elif state == RState.DRIVING and spd < 5:
            desc   = "stationary"

        # lane name
        ln = ["slow lane","fast lane","shoulder"][lane] if lane < 3 else f"lane {lane}"

        # Player prefix
        if player:
            mid = _player_machine_id(car)
            label_str = f"[{style_c}]{emoji} {label}[/{style_c}] [dim]P{mid}·{ln}·{spd:.0f}km/h[/dim]"
        elif remote:
            label_str = f"[dim cyan]{emoji} {label}↗[/dim cyan] [dim]{ln}·{spd:.0f}km/h[/dim]"
        else:
            label_str = f"[{style_c}]{emoji} {label}[/{style_c}] [dim]{ln}·{spd:.0f}km/h[/dim]"

        lines.append_text(Text.from_markup(
            f" {label_str}  [{style}]{icon} {desc}[/{style}]\n"))

    if not active:
        lines.append("  🏁 All cars have finished the route\n", style="bold green")

    return Panel(
        lines,
        title="[bold green]🪪 Car Profiles — live state[/bold green]",
        subtitle="[dim]one line per car · what each car is doing right now[/dim]",
        border_style="green",
        padding=(0,0),
    )

_EVENT_ICONS = {
    "WEATHER":    ("🌤", "cyan"),
    "COLLISION":  ("💥", "bold red"),
    "NEAR_MISS":  ("⚠",  "bold yellow"),
    "FUEL_OUT":   ("⛽", "bold red"),
    "PREEMPT":    ("🚨", "bold red"),
    "CORRIDOR":   ("🚨", "bold magenta"),
    "EMERG":      ("🚨", "bold red"),
    "BRAKE":      ("⛔", "red"),
    "HAZARD":     ("⚠",  "yellow"),
    "SCHOOL":     ("🏫", "bright_yellow"),
    "ZONE":       ("🏫", "bright_yellow"),
    "ROADWORK":   ("🚧", "red"),
    "CONVOY":     ("🚛", "green"),
    "PLATOON":    ("🚛", "green"),
    "SPLIT":      ("🔌", "magenta"),
    "RECONNECT":  ("🔌", "cyan"),
    "OVERTAKE":   ("↑",  "bold green"),
    "LANE_CHANGE":("→",  "cyan"),
    "CHAIN":      ("📡", "cyan"),
    "SMART":      ("📡", "cyan"),
    "PREDICTED":  ("🧠", "bold cyan"),
}

def _event_style(etype, detail):
    combined = (etype + " " + detail).upper()
    for key, (icon, style) in _EVENT_ICONS.items():
        if key in combined:
            return icon, style
    return "·", "dim white"

def _build_events_panel(events, max_lines=22):
    recent = events[-max_lines:] if events else []
    lines  = []
    for ev in reversed(recent):
        etype  = str(ev.get("event",""))
        detail = str(ev.get("detail", ev.get("event","")))[:85]
        t      = time.strftime("%H:%M:%S", time.localtime(ev.get("ts", time.time())))
        icon, style = _event_style(etype, detail)
        lines.append(f"[dim]{t}[/dim] [{style}]{icon} {detail}[/{style}]")

    content = "\n".join(lines) if lines else "[dim]Waiting for V2X events...[/dim]"
    return Panel(
        content,
        title="[bold cyan]📡 V2X Events  (newest first)[/bold cyan]",
        subtitle="[dim]radio messages, warnings, incidents[/dim]",
        border_style="cyan",
        padding=(0,1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROLS BAR  (always visible at bottom)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_controls_bar():
    t = Text()
    t.append("  ⌨  DRIVE: ", style="bold dim")
    pairs = [("L","left lane"), ("R","right lane"), ("O","overtake"),
             ("B","brake"), ("A","accelerate"), ("Y","yield"), ("N","resume")]
    for key, desc in pairs:
        t.append(f"[{key}]", style="bold cyan")
        t.append(f"={desc}  ", style="dim")
    t.append("│  Ctrl+C = quit", style="dim")
    return Panel(t, border_style="dim", padding=(0,0))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ═══════════════════════════════════════════════════════════════════════════════

def render_road(status, events, terminal_width=None):
    tw      = terminal_width or console.width or 130
    strip_w = max(40, tw - 18)   # 18 chars for lane label prefix

    layout = Layout()
    layout.split_column(
        Layout(name="road",     ratio=4),
        Layout(name="middle",   ratio=5),
        Layout(name="profile",  ratio=3),
        Layout(name="controls", size=3),
    )
    layout["middle"].split_row(
        Layout(name="events",   ratio=5),
        Layout(name="activity", ratio=4),
    )

    layout["road"].update(_build_road_panel(status, strip_w))
    layout["events"].update(_build_events_panel(events))
    layout["activity"].update(_build_activity_panel(status))
    layout["profile"].update(_build_car_profile_panel(status))
    layout["controls"].update(_build_controls_bar())

    return layout


# ── Dashboard driver ──────────────────────────────────────────────────────────

class RoadDashboard:
    def __init__(self, sim, refresh_rate=5):
        self.sim          = sim
        self.refresh_rate = refresh_rate
        self._running     = False

    def run(self):
        self._running = True
        with Live(console=console, refresh_per_second=self.refresh_rate,
                  screen=True) as live:
            while self._running and not self.sim.is_all_done():
                try:
                    status = self.sim.get_status()
                    events = self.sim.get_token_events()
                    tw     = console.width or 130
                    live.update(render_road(status, events, tw))
                except Exception as e:
                    live.update(Panel(f"[red]Dashboard error: {e}[/red]"))
                time.sleep(1.0 / self.refresh_rate)

    def stop(self):
        self._running = False
