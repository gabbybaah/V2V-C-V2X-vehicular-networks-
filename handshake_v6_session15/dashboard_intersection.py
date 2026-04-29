# =============================================================================
# dashboard_intersection.py  —  Road-sim style intersection dashboard
# =============================================================================
import time
from rich.console import Console
from rich.live    import Live
from rich.text    import Text
from rich.panel   import Panel
from rich.layout  import Layout
from rich.table   import Table
from rich         import box as rbox

console = Console()

# ── Colours matching road sim ─────────────────────────────────────────────────
_CSTY = {
    "SMART":     "bright_yellow",
    "LEGACY":    "white",
    "EMERGENCY": "bold bright_red",
    "ROGUE":     "bold bright_magenta",
    "BREAKDOWN": "dim yellow",
    "PLATOON":   "bright_magenta",
}
_TOK_STY    = "bold bright_green"
_EMERG_GLYPH = "🚨"

def _csty(car):
    ct = str(car.get("type", car.get("car_type","SMART"))).upper()
    if car.get("token_holder"): return _TOK_STY
    return _CSTY.get(ct, "bright_yellow")

def _is_emerg(car):
    return str(car.get("type", car.get("car_type",""))).upper() == "EMERGENCY"

_ARROW  = "►"   # fallback — coloured by type
_ARM_ARROW = {   # each arm's cars point TOWARD the box center
    "North": "↓",
    "South": "↑",
    "East":  "←",
    "West":  "→",
}
_EXIT_ARM  = {
    "North":{"STRAIGHT":"South","TURN_RIGHT":"East","TURN_LEFT":"West"},
    "South":{"STRAIGHT":"North","TURN_RIGHT":"West","TURN_LEFT":"East"},
    "East": {"STRAIGHT":"West", "TURN_RIGHT":"South","TURN_LEFT":"North"},
    "West": {"STRAIGHT":"East", "TURN_RIGHT":"North","TURN_LEFT":"South"},
}

def _phase_style(ph):
    return {"NS_GREEN":"bold green","NS_YELLOW":"bold yellow",
            "EW_GREEN":"bold green","EW_YELLOW":"bold yellow",
            "ALL_RED":"bold red","ALL_RED_2":"bold red",
            "PREEMPTED":"bold magenta","PEDESTRIAN":"bold cyan"}.get(ph,"white")

def _signal_bar(rem, w=14):
    f = max(0,min(w,int(rem/30*w)))
    return "█"*f+"░"*(w-f)

def _arm_light(arm, ga, cor, ped):
    if cor:        return "🚨","bold magenta"
    if ped:        return "🚶","bold cyan"
    if arm in ga:  return "🟢","bold green"
    return "🔴","bold red"

# ── Canvas geometry ───────────────────────────────────────────────────────────
CW        = 72
BOX_L     = 20
BOX_INNER = 30
BOX_R     = BOX_L + 1 + BOX_INNER   # 51
W_W       = BOX_L                    # 20
E_W       = CW - BOX_R - 1          # 20
NS_OUT    = BOX_L + BOX_INNER//2 - 2  # 33
NS_DIV    = NS_OUT + 2               # 35
NS_IN     = NS_DIV + 2               # 37

def _row(out, t):
    out.append_text(t); out.append("\n")

def _term_w(s):
    """Terminal column width of a string — wide chars (emoji) count as 2."""
    return sum(2 if ord(ch) > 0xFFFF else 1 for ch in s)

def _ns_slot_row(car, arm, out_ind):
    t = Text()
    t.append(" "*NS_OUT, style="dim")
    t.append(out_ind, style="dim")
    t.append(" ")
    t.append("│", style="dim")
    t.append(" ")
    if car is not None:
        sty = _csty(car)
        lbl = car.get("label","?")[:4]
        if _is_emerg(car):
            t.append(_EMERG_GLYPH, style=sty)
            t.append(lbl, style=sty)
        elif car.get("token_holder", False):
            t.append("★" + lbl, style=sty)
        else:
            t.append(_ARM_ARROW[arm] + lbl, style=sty)
    return t

def _w_arm_strip(wq, light_ic, light_sty):
    """W arm: 20 chars. Light at left, 4 car slots (4 chars each).
    wq sorted CLOSEST→furthest. Display right-justified so closest is always
    rightmost (adjacent to box wall). Empty slots fill from the left."""
    t = Text()
    t.append(light_ic, style=light_sty)   # 2 cols
    t.append(" ")                          # 1 col  → 3 used
    slot_w  = 4
    n_slots = (W_W - 3) // slot_w         # 4
    cars    = wq[:n_slots]                 # up to 4 closest
    # Right-justify: Nones on left, then furthest-of-shown→closest left→right
    n_empty = n_slots - len(cars)
    slots   = [None] * n_empty + list(reversed(cars))
    for car in slots:
        if car is None:
            t.append("·" + "─" * (slot_w - 1), style="dim")
        elif _is_emerg(car):
            t.append(_EMERG_GLYPH, style=_csty(car))
            t.append(" " * (slot_w - 2))
        else:
            sty   = _csty(car)
            lbl   = car.get("label","?")[:3]
            glyph = "★" if car.get("token_holder", False) else _ARM_ARROW["West"]  # → toward center
            t.append(glyph + lbl, style=sty)
            t.append(" " * (slot_w - 1 - len(lbl)))
    return t

def _e_arm_strip(eq, light_ic, light_sty):
    """E arm: 20 chars. 4 car slots then light at right.
    eq sorted CLOSEST→furthest. Display left-justified so closest is always
    leftmost (adjacent to box wall). Empty slots fill from the right."""
    t = Text()
    slot_w  = 4
    n_slots = (E_W - 3) // slot_w   # 4
    cars    = eq[:n_slots]           # up to 4 closest, index 0 = closest
    slots   = cars + [None] * (n_slots - len(cars))   # left-justified
    for car in slots:
        if car is None:
            t.append("─" * (slot_w - 1) + "·", style="dim")
        elif _is_emerg(car):
            t.append(_EMERG_GLYPH, style=_csty(car))
            t.append(" " * (slot_w - 2))
        else:
            sty   = _csty(car)
            lbl   = car.get("label","?")[:3]
            glyph = "★" if car.get("token_holder", False) else _ARM_ARROW["East"]  # ← toward center
            t.append(glyph + lbl, style=sty)
            t.append(" " * (slot_w - 1 - len(lbl)))
    t.append(" ")
    t.append(light_ic, style=light_sty)
    return t


def _build_canvas(status, arms, ga):
    cor   = status.get("corridor_active",False)
    ped   = status.get("pedestrian",False)
    phase = status.get("light_phase","")
    rem   = status.get("light_remaining",0)
    done  = status.get("done",0); total = status.get("total_cars",0)
    el    = status.get("elapsed_s",0)
    tput  = done/el*60 if el>3 and done>0 else 0
    nmiss = status.get("box_near_miss_count",0)
    ps    = _phase_style(phase)

    n_ic,n_sty = _arm_light("North",ga,cor,ped)
    s_ic,s_sty = _arm_light("South",ga,cor,ped)
    e_ic,e_sty = _arm_light("East", ga,cor,ped)
    w_ic,w_sty = _arm_light("West", ga,cor,ped)

    # Queues — closest to stop line first
    nq = sorted(arms.get("North",{}).get("queue_cars",[]), key=lambda c: c.get("dist",0))[:5]
    sq = sorted(arms.get("South",{}).get("queue_cars",[]), key=lambda c: c.get("dist",0))[:5]
    # W arm: closest first — _w_arm_strip reverses them so closest is rightmost (at box)
    wq = sorted(arms.get("West",{}).get("queue_cars",[]),  key=lambda c: c.get("dist",0))[:4]
    # E arm: closest first — displayed left→right, closest leftmost (at box)
    eq = sorted(arms.get("East",{}).get("queue_cars",[]),  key=lambda c: c.get("dist",0))[:4]

    # Crossing cars
    xing = []
    for an, ad in arms.items():
        for car in ad.get("road_cars",[]):
            man  = car.get("maneuver","STRAIGHT")
            dest = car.get("dest_arm", _EXIT_ARM.get(an,{}).get(man,"South"))
            tok  = car.get("token_holder",False)
            ct   = str(car.get("type",car.get("car_type","SMART"))).upper()
            sty  = _TOK_STY if tok else _CSTY.get(ct,"bright_yellow")
            xing.append({"entry":an,"dest":dest,"label":car.get("label","?"),
                         "sty":sty,"tok":tok,"progress":car.get("progress",0.0),
                         "man":man,"is_emerg":ct=="EMERGENCY"})

    out = Text()

    # ── Signal bar
    bar = _signal_bar(rem)
    _row(out, Text.from_markup(
        f" [{ps}]{phase}[/{ps}]  [dim][{bar}] {rem:.0f}s[/dim]"
        f"   [{n_sty}]N:{n_ic}[/{n_sty}]"
        f"  [{s_sty}]S:{s_ic}[/{s_sty}]"
        f"  [{e_sty}]E:{e_ic}[/{e_sty}]"
        f"  [{w_sty}]W:{w_ic}[/{w_sty}]"
        + (f"   [bold yellow]⚠ {nmiss} near-miss[/bold yellow]" if nmiss else "")
        + ("   [bold magenta]🚨 EMERGENCY CORRIDOR — ALL YIELD[/bold magenta]" if cor else "")
        + ("   [bold cyan]🚶 PEDESTRIAN CROSSING — ALL RED[/bold cyan]" if ped else "")
    ))
    out.append("\n")

    # ── North arm
    t = Text(" "*NS_IN); t.append(f"N:{n_ic}", style=n_sty); _row(out, t)
    n_disp = max(3, len(nq))
    for r in range(n_disp):
        slot_idx = n_disp - 1 - r   # top row = furthest
        car      = nq[slot_idx] if slot_idx < len(nq) else None
        out_ind  = "↑" if r == 0 else " "
        _row(out, _ns_slot_row(car, "North", out_ind))

    # ── Top stop line / box border
    bc_top = ("bold green" if "North" in ga and not cor and not ped
              else "bold magenta" if cor else "bold cyan" if ped else "bold red")
    _row(out, Text.from_markup(
        f"[dim]{'─'*BOX_L}[/dim][{bc_top}]╔{'═'*BOX_INNER}╗[/{bc_top}][dim]{'─'*E_W}[/dim]"
    ))

    # ── E/W arms + box interior
    box_rows = max(3, len(xing)+1)
    for br in range(box_rows):
        t = Text()
        # West side
        if br == 0:
            t.append_text(_w_arm_strip(wq, w_ic, w_sty))
        elif br == 1:
            t.append(" "*(W_W-2), style="dim"); t.append("←─", style="dim")
        else:
            t.append("─"*W_W, style="dim")
        t.append("║", style="dim cyan")

        # Box interior — MUST be exactly BOX_INNER terminal columns wide
        if br < len(xing):
            c    = xing[br]
            lbl  = c["label"][:5]; sty = c["sty"]
            prog = c["progress"]
            dest = c["dest"]
            b10  = "█"*int(prog*10)+"░"*(10-int(prog*10))
            pct_str = f" {int(prog*100):3}% "   # always 6 chars: " NNN% "
            inner = Text()
            pw = 0   # track terminal column width
            inner.append(" "); pw += 1
            if c["is_emerg"]:
                inner.append(_EMERG_GLYPH, style=sty); pw += 2   # emoji = 2 cols
            elif c["tok"]:
                inner.append("★", style=sty); pw += 1
            else:
                inner.append(_ARROW, style=sty); pw += 1
            inner.append(lbl, style=sty); pw += _term_w(lbl)
            inner.append(f"[{b10}]", style=sty); pw += 12        # [ + 10 blocks + ]
            inner.append(dest[0], style="dim"); pw += 1           # exit arm initial
            inner.append(pct_str, style="dim"); pw += len(pct_str)
            # Pad to exactly BOX_INNER
            if pw < BOX_INNER:
                inner.append(" " * (BOX_INNER - pw))
        elif br == box_rows-1:
            cx  = ("[bold magenta]╬[/bold magenta]" if cor else
                   "[bold cyan]╬[/bold cyan]"       if ped  else
                   "[bold white]╬[/bold white]")
            pad = (BOX_INNER-1)//2
            inner = Text.from_markup(" "*pad + cx + " "*(BOX_INNER-1-pad))
        else:
            inner = Text(" "*BOX_INNER)
        t.append_text(inner)

        t.append("║", style="dim cyan")
        # East side
        if br == 0:
            t.append_text(_e_arm_strip(eq, e_ic, e_sty))
        elif br == 1:
            t.append("─→", style="dim"); t.append(" "*(E_W-2), style="dim")
        else:
            t.append("─"*E_W, style="dim")
        _row(out, t)

    # ── Bottom stop line / box border
    bc_bot = ("bold green" if "South" in ga and not cor and not ped
              else "bold magenta" if cor else "bold cyan" if ped else "bold red")
    _row(out, Text.from_markup(
        f"[dim]{'─'*BOX_L}[/dim][{bc_bot}]╚{'═'*BOX_INNER}╝[/{bc_bot}][dim]{'─'*E_W}[/dim]"
    ))

    # ── South arm
    s_disp = max(3, len(sq))
    for r in range(s_disp):
        car     = sq[r] if r < len(sq) else None
        out_ind = "↓" if r == s_disp-1 else " "
        _row(out, _ns_slot_row(car, "South", out_ind))
    t = Text(" "*NS_IN); t.append(f"S:{s_ic}", style=s_sty); _row(out, t)
    out.append("\n")

    # ── Footer
    pct2 = done/total*100 if total else 0
    out.append_text(Text.from_markup(
        f" [dim]✓ {done}/{total} cleared ({pct2:.0f}%)   📊 {tput:.1f}/min   "
        f"[{_TOK_STY}]★[/{_TOK_STY}]=token  "
        f"[bright_yellow]↓[/bright_yellow]=N  "
        f"[bright_yellow]↑[/bright_yellow]=S  "
        f"[bright_yellow]→[/bright_yellow]=W  "
        f"[bright_yellow]←[/bright_yellow]=E  "
        f"🚨=emergency[/dim]"
    ))
    return out


# ── Header ────────────────────────────────────────────────────────────────────
def _build_header(status):
    el=status.get("elapsed_s",0); total=status.get("total_cars",0)
    done=status.get("done",0);    pct=status.get("pct_done",0)
    phase=status.get("light_phase",""); tput=status.get("throughput_rate",0)
    nmiss=status.get("box_near_miss_count",0)
    cor=status.get("corridor_active",False); ped=status.get("pedestrian",False)
    ps=_phase_style(phase)
    h=Text()
    h.append("HANDSHAKE V6 · INTERSECTION  ",style="bold blue")
    h.append(f"T+{el:.0f}s  ",style="cyan")
    h.append(f"✓ {done}/{total} ({pct:.0f}%)  ",style="bold green")
    h.append(phase+"  ",style=ps)
    if tput>0: h.append(f"📊 {tput:.1f}/min  ",style="green")
    if nmiss:  h.append(f"⚠ {nmiss} near-miss  ",style="bold yellow")
    if cor:    h.append("🚨 CORRIDOR  ",style="bold red")
    if ped:    h.append("🚶 PEDESTRIAN  ",style="bold cyan")
    return Panel(h,border_style="blue",padding=(0,1))


# ── Event log ─────────────────────────────────────────────────────────────────
_EV={
    "CROSSED":           ("[CLEARED]",    "bold green"),
    "TOKEN_ISSUED":      ("[TOKEN]",      "bold bright_cyan"),
    "TOKEN_ACK":         ("[ACK]",        "cyan"),
    "TOKEN_CANCEL":      ("[CANCEL]",     "yellow"),
    "NEG_RECV":          ("[NEG REQ]",    "dim yellow"),
    "NEG_REQUEST_SENT":  ("[NEG SENT]",   "dim yellow"),
    "PREEMPT":           ("[EMERG]",      "bold red"),
    "PREEMPT_BROADCAST": ("[EMERG TX]",   "bold red"),
    "AMBULANCE_CORRIDOR":("[CORRIDOR]",   "bold magenta"),
    "YIELDING":          ("[YIELD]",      "magenta"),
    "YIELD_DONE":        ("[RESUME]",     "cyan"),
    "PEDESTRIAN":        ("[PED XING]",   "bold cyan"),
    "ROGUE_VIOLATION":   ("[ROGUE!]",     "bold red"),
    "MACHINE_FAIL":      ("[ARM DOWN]",   "bold red"),
    "MACHINE_RECOVER":   ("[ARM BACK]",   "bold green"),
    "NEAR_MISS_BOX":     ("[NEAR-MISS]",  "bold yellow"),
    "SLOT_CONFLICT":     ("[CONFLICT]",   "bold red"),
    "CONFLICT_RESOLVED": ("[RESOLVED]",   "cyan"),
    "LIGHT_CHANGE":      ("[SIGNAL]",     "blue"),
    "TOKEN_RECEIVED":    ("[RECV]",       "cyan"),
    "ROW_NEGOTIATION":   ("[NEGOTIATE]",  "yellow"),
}
def _build_event_log(events, max_lines=22):
    recent=events[-max_lines:] if events else []
    lines=[]; now=time.time()
    for ev in reversed(recent):
        etype=str(ev.get("event",""))
        detail=str(ev.get("detail",ev.get("phase","")))
        arm=str(ev.get("arm","ALL"))[:5]; age=now-ev.get("ts",now)
        lbl,sty=_EV.get(etype,(f"[{etype[:12]}]","dim white"))
        arm_tag=f"[dim]{arm:>5}[/dim]" if arm!="ALL" else "     "
        lines.append(f"[dim]{age:.0f}s[/dim] {arm_tag} [{sty}]{lbl}[/{sty}] [dim]{detail[:48]}[/dim]")
    content="\n".join(lines) if lines else "[dim]Waiting for events...[/dim]"
    return Panel(content,title="[bold yellow]📋 Events[/bold yellow]",
                 subtitle="[dim]newest first[/dim]",border_style="yellow",padding=(0,1))


# ── Token log ─────────────────────────────────────────────────────────────────
_TE={
    "TOKEN_ISSUED":      ("bold bright_green","+","ISSUED"),
    "TOKEN_ACK":         ("bold cyan",        "✓","ACK"),
    "TOKEN_CANCEL":      ("bold red",         "✗","CANCEL"),
    "NEG_RECV":          ("dim yellow",       "?","NEG REQ"),
    "NEG_REQUEST_SENT":  ("dim yellow",       "?","NEG SENT"),
    "TOKEN_RECEIVED":    ("cyan",             "↓","RECV"),
    "PREEMPT":           ("bold magenta",     "!","PREEMPT"),
    "PREEMPT_BROADCAST": ("bold magenta",     "!","PREEMPT TX"),
    "AMBULANCE_CORRIDOR":("bold magenta",     "!","CORRIDOR"),
    "CROSSED":           ("bold green",       "✓","CROSSED"),
    "ROGUE_VIOLATION":   ("bold red",         "⚠","ROGUE"),
    "SLOT_CONFLICT":     ("bold red",         "!","CONFLICT"),
    "CONFLICT_RESOLVED": ("cyan",             "✓","RESOLVED"),
    "LIGHT_CHANGE":      ("bold blue",        "L","LIGHT"),
    "PEDESTRIAN":        ("bold cyan",        "🚶","PED XING"),
    "NEAR_MISS_BOX":     ("bold yellow",      "⚠","NEAR-MISS"),
    "MACHINE_FAIL":      ("bold red",         "X","ARM FAIL"),
    "MACHINE_RECOVER":   ("bold green",       "✓","ARM BACK"),
}
def _build_token_panel(events, height=18):
    tbl=Table("Ago","Arm","Ev","Detail",box=rbox.SIMPLE,show_header=True,
              header_style="bold dim",padding=(0,1),expand=True)
    tbl.columns[0].width=4;tbl.columns[1].width=5
    tbl.columns[2].width=10;tbl.columns[3].ratio=1
    now=time.time()
    for ev in events[-height:]:
        etype=str(ev.get("event",""))
        detail=str(ev.get("detail",ev.get("phase","")))
        arm=str(ev.get("arm","ALL"))[:5]
        age=f"{now-ev.get('ts',now):.0f}s"
        sty,icon,desc=_TE.get(etype,("dim white","·",etype[:10]))
        tbl.add_row(f"[dim]{age}[/dim]",f"[dim]{arm}[/dim]",
                    f"[{sty}]{icon} {desc}[/{sty}]",
                    f"[{sty}]{detail[:42]}[/{sty}]")
    return Panel(tbl,title="[bold yellow]📡 Token Exchange[/bold yellow]",
                 subtitle="[dim]NEG→ISSUED→ACK→CROSSED[/dim]",border_style="yellow")


# ── Stats — two-column layout to keep height minimal ─────────────────────────
def _build_stats_panel(status):
    total=status.get("total_cars",0);done=status.get("done",0)
    queued=status.get("queued",0);road=status.get("on_road",0)
    el=status.get("elapsed_s",0);pct=status.get("pct_done",0)
    tput=status.get("throughput_rate",0);peak=status.get("peak_throughput",0)
    rush=status.get("rush_hour",False);unctrl=status.get("uncontrolled",False)
    cor=status.get("corridor_active",False);ped=status.get("pedestrian",False)
    dead=status.get("dead_arms",[]);nmiss=status.get("box_near_miss_count",0)

    # Build two parallel lists of (key, value markup) — displayed side by side
    left_rows = []
    right_rows = []

    # Status flags (left)
    if rush:   left_rows.append(("MODE","[bold yellow]RUSH HOUR[/bold yellow]"))
    if unctrl: left_rows.append(("MODE","[bold cyan]UNCONTROLLED[/bold cyan]"))
    if cor:    left_rows.append(("","[bold red]🚨 CORRIDOR[/bold red]"))
    if ped:    left_rows.append(("","[bold cyan]🚶 PED XING[/bold cyan]"))
    for da in dead: left_rows.append(("",f"[bold red]{da} DOWN[/bold red]"))

    left_rows += [
        ("Fleet",    f"[bold]{total}[/bold]"),
        ("Cleared",  f"[bold green]{done}/{total}[/bold green]"),
        ("Progress", f"[bold green]{pct:.0f}%[/bold green]"),
        ("Queued",   f"[yellow]{queued}[/yellow]"),
        ("In box",   f"[cyan]{road}[/cyan]"),
        ("Runtime",  f"[cyan]{el:.0f}s[/cyan]"),
    ]
    if tput>0: left_rows.append(("Tput", f"[bold green]{tput:.1f}/min[/bold green]"))
    if peak>0: left_rows.append(("Peak", f"[green]{peak:.1f}/min[/green]"))
    if el>5 and done>0 and done<total:
        left_rows.append(("ETA", f"[dim]~{(total-done)/(done/el):.0f}s[/dim]"))
    if nmiss: left_rows.append(("⚠",f"[bold yellow]{nmiss} near-miss[/bold yellow]"))

    right_rows += [
        ("Protocol", "[dim]Handshake V6[/dim]"),
        ("CVC",      "[dim]21450 21806[/dim]"),
        ("─────", ""),
        (f"[{_TOK_STY}]★[/{_TOK_STY}]",   "[dim]token holder[/dim]"),
        ("🚨",                               "[dim]emergency[/dim]"),
        (f"[bright_yellow]↓[/bright_yellow]", "[dim]N arm (toward ╬)[/dim]"),
        (f"[bright_yellow]↑[/bright_yellow]", "[dim]S arm[/dim]"),
        (f"[bright_yellow]→[/bright_yellow]", "[dim]W arm[/dim]"),
        (f"[bright_yellow]←[/bright_yellow]", "[dim]E arm[/dim]"),
    ]

    # Pad to equal length so the 2-col table renders evenly
    max_len = max(len(left_rows), len(right_rows))
    left_rows  += [("","")] * (max_len - len(left_rows))
    right_rows += [("","")] * (max_len - len(right_rows))

    tbl = Table(box=None, show_header=False, padding=(0,1), expand=True)
    tbl.add_column("K1", style="dim", width=9)
    tbl.add_column("V1", ratio=1)
    tbl.add_column("K2", style="dim", width=9)
    tbl.add_column("V2", ratio=1)

    for (k1,v1),(k2,v2) in zip(left_rows, right_rows):
        tbl.add_row(k1, v1, k2, v2)

    return Panel(tbl, title="[bold]📊 Stats[/bold]", border_style="dim", padding=(0,0))


# ── Render — left=canvas+stats, right=events+tokens ─────────────────────────
def render(status, token_events):
    arms=status.get("arms",{})
    ga=status.get("light_green",[])
    cor=status.get("corridor_active",False)
    ped=status.get("pedestrian",False)
    bdr=("bold magenta" if cor else "bold cyan" if ped else "cyan")
    phase=status.get("light_phase","")
    canvas_panel=Panel(
        _build_canvas(status,arms,ga),
        title=f"[bold cyan]╬  INTERSECTION[/bold cyan]  [dim]{phase}[/dim]",
        subtitle="[dim]fixed crossroads · [bright_yellow]↓↑→←[/bright_yellow]=toward ╬ · colour=[bright_yellow]smart[/bright_yellow] [white]legacy[/white] [bold bright_magenta]rogue[/bold bright_magenta] · 🚨=emerg ★=token[/dim]",
        border_style=bdr,padding=(0,1),
    )
    layout=Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main",   ratio=1),
    )
    layout["main"].split_row(
        Layout(name="left",  ratio=3),
        Layout(name="right", ratio=2),
    )
    # Left column: canvas fills top, compact 2-col stats sits below
    layout["left"].split_column(
        Layout(name="canvas", ratio=1),
        Layout(name="stats",  size=11),
    )
    # Right column: events on top, tokens below — both get more room now
    layout["right"].split_column(
        Layout(name="events", ratio=3),
        Layout(name="tokens", ratio=2),
    )
    layout["header"].update(_build_header(status))
    layout["canvas"].update(canvas_panel)
    layout["stats"].update(_build_stats_panel(status))
    layout["events"].update(_build_event_log(token_events))
    layout["tokens"].update(_build_token_panel(token_events))
    return layout


class IntersectionDashboard:
    def __init__(self, sim):
        self.sim=sim; self._running=False
    def run(self):
        self._running=True
        with Live(console=console,refresh_per_second=5,screen=True) as live:
            while self._running:
                try:
                    status=self.sim.get_status()
                    toks=self.sim.get_token_events()
                    live.update(render(status,toks))
                    if status.get("all_done"):
                        time.sleep(5); break
                except Exception: pass
                time.sleep(0.2)
    def stop(self):
        self._running=False
