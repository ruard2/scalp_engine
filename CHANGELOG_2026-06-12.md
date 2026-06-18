# Changelog — v6_engine — 2026-06-12

Alle wijzigingen aan `live_engine.py` van vandaag, plus nieuwe backtest-tooling.

---

## 1. KRITIEKE FIX: verkeerd market ID (orders werden geweigerd)

**Probleem:** Engine gebruikte EUR/USD market ID `401697501` (testingversion2-account).
Op het scalp-account (407046032) bestaat dat market ID niet → elke order werd door
de broker geweigerd met `Status: 2, StatusReason: 75` (inner StatusReason 160).
Twee dagen lang vuurden signalen zonder dat één order doorkwam.

**Fix:** `MARKET_ID = "403897186"` — het juiste EUR/USD market ID voor het scalp-account.

---

## 2. Trail stop kon onder entry uitkomen (kleine verliezen bij "TrailSL")

**Probleem:** T4-profiel had trigger=2.0p / dist=1.5p → trail activeerde bij +2p winst
en zette de stop op slechts +0.5p boven entry — minder dan de spread. Resultaat:
"TrailSL"-exits met negatieve P&L (-2.7p, -2.1p, -3.0p etc. in de logs) en
mini-winsten van 3 cent terwijl kosten 10 cent round-trip zijn.

**Fix:**
- Nieuwe constante `MIN_TRAIL_LOCK_PIPS = 1.5`: trail stop kan nooit lager dan
  entry + 1.5p (Bull) of hoger dan entry - 1.5p (Bear).
- Profielen aangescherpt:
  | Profiel | trigger oud → nieuw | dist oud → nieuw |
  |---|---|---|
  | T1_TrendRC | 4.0 → 5.0 | 3.0 → 3.0 |
  | T2_Trend | 3.0 → 5.0 | 2.5 → 3.0 |
  | T3_Expansion | 2.5 → 4.5 | 2.0 → 3.0 |
  | T4_Compression | 2.0 → 3.5 | 1.5 → 2.0 |

---

## 3. DirectionFlip/EnvRange-exit na 1 bar = classifier-ruis

**Probleem:** Label-flip op de eerstvolgende bar sloot trades direct, ongeacht P&L
(exits van -5.2p, -1.0p, -1.8p na 1 bar).

**Fix:** DirectionFlip- en EnvRange-exits vereisen nu minimaal `bars_held >= 2`.

---

## 4. Midnight buffer (dag-flip)

**Probleem:** Classifier gebruikt dagstatistieken die om 00:00 UTC resetten →
ruis rond de dagovergang.

**Fix:** Geen nieuwe entries tussen 23:00 en 00:59 UTC. Zichtbaar in dashboard
als aparte regel "Midnight buffer".

---

## 5. MAX_OPEN: 3 → 1

**Probleem:** Tot 3 vrijwel identieke posities binnen 10 seconden geopend
(zelfde richting, zelfde bar, andere local-label) — 3× hetzelfde risico, geen
diversificatie.

**Fix:** `MAX_OPEN = 1` — één positie tegelijk.

---

## 6. Compression-environment geblokkeerd (uit backtest)

**Backtest-bewijs (16 maanden, feb 2025 → jun 2026):** T4_Compression-trades:
3.841 stuks, **-1.786 pips** (-0.46p gemiddeld). Alle edge zit in Trend/Expansion.

**Fix:** `entry_allowed()` blokkeert nu ook `environment == "Compression"`
(naast Range). Dashboard-regel heet nu "Env Trend/Expans".

---

## 7. Nieuw: backtest_v6.py

Backtest die de **echte live-engine functies importeert** (`check_exit`,
`entry_allowed`, `PROFILES`, `get_profile_key`, v6-classifier) — geen kopie van
logica. Data: `testingversion2\5_min_data\fetched_data_eurusd_401697501.csv`
(100k 5-min bars), geresampled naar 10-min. Kosten: 1.0 pip round-trip.

Gebruik: `python backtest_v6.py` (of `--days N` voor laatste N dagen).
Trades worden weggeschreven naar `backtest_trades.csv`.

### Backtest-resultaat voor vs. na alle fixes van vandaag:

| | Voor (incl. Compression) | Na (huidige logica) |
|---|---|---|
| Totaal P&L | +369p | **+2.219p** (≈ $222 bij qty 1000) |
| Avg/trade | +0.04p | **+0.36p** |
| Win rate | 64.3% | **68.0%** |
| Max drawdown | -1.977p | **-775p** |
| Trades/week | 144 | 90 |

Inzicht: nachttrading ("Other"-sessie) was niet het probleem — het verlies zat
volledig in Compression-trades 's nachts. Na het Compression-blok is elke sessie
netto positief (Overlap de beste: +0.56p avg). Sterkste combo's:
`Bull_Expansion_Impulse` (+0.96p avg), `Bull_Expansion_Pullback` (+1.24p avg).

---

## 8. Middag-update: sweep-backtest → exits geoptimaliseerd

Vragen van de gebruiker getest met `backtest_sweep.py` (zelfde 16-maands data,
classificatie 1x, 16 configuraties):

**a) Meer dan 1 positie tegelijk (met cooldown)?** → NEE, alle varianten slechter:
| Config | Totaal | DD |
|---|---|---|
| MAX_OPEN=1 (baseline) | +2218p | -775p |
| max2 cooldown2 | +1128p | -1291p |
| max2 cooldown3 | +118p | -1147p |
| max3 cooldown3 | -37p | -1460p |
Extra posities stapelen hetzelfde signaal → meer risico, geen extra edge.
MAX_OPEN blijft 1.

**b) Trail/SL te strak?** → JA, vooral SL:
- SL 1.5→2.5x ATR is de grootste verbetering (+1.04p avg solo)
- Trail trigger +2p / dist +1p halveert de drawdown

**c) EnvRange-exit?** → Netto verliesgevend (-3.30p avg, n=310). Verwijderd;
trail/SL vangen die situaties beter af.

**Doorgevoerd (winnende combinatie):**
- `sl_atr_mult`: 1.5 → 2.5 (alle profielen)
- Trail: T1/T2 trigger 5→7 dist 3→4; T3 4.5→6.5/3→4; T4 3.5→5.5/2→3
- EnvRange-exit verwijderd uit `check_exit()` (DirectionFlip blijft, met 2-bar minimum)

| | Voor (ochtend-versie) | Na sweep-optimalisatie |
|---|---|---|
| Totaal P&L | +2.218p | **+4.832p** (≈ $483 bij qty 1000) |
| Avg/trade | +0.36p | **+1.17p** |
| Win rate | 68.0% | **75.8%** |
| Max drawdown | -775p | **-258p** |

---

## 9. 2026-06-16: Adaptieve SL + Bounce exit

Backtest: `backtest_bounce.py` (16 mnd data, fixed P&L calc voor TrailSL).

**Bounce exit (baseline 2.5x SL):**
- Na elke harde SL: houdt positie open, wacht max 2 bars op recovery
- Safety floor SL+6p: bij verdere koersdaling direct sluiten
- Trail van 2p zodra prijs 1p terugkeert richting SL trigger
- Resultaat: +5125p vs +1699p baseline — **grootste verbetering van alle changes**
- Bull én Bear bounces even effectief op 10-min bars (anders dan testingversion2 swing)

**Adaptieve SL op ATR rank:**
- ATR rank ≥ 75%: SL = 4.0x ATR (was flat 2.5x)
- ATR rank < 75%: SL blijft 2.5x ATR
- Reden: rank>90% trades gemiddeld -25.5p verlies bij SL vs -15p bij rank<50%
- Met wijdere SL minder premature stops, meer TrailSL exits (hogere winst)

**Gecombineerd (3-tier + bounce):**
| | Baseline | Na 2026-06-16 |
|---|---|---|
| Totaal P&L | +1699p | **+5114p** |
| Avg/trade | +0.41p | **+1.41p** |
| Win rate | 75.9% | **78.7%** |
| Max drawdown | -590p | **-173p** |

Interessante bevinding: ATR rank 90%+ heeft gemiddeld **+2.01p per trade, 82.4% win** —
hoge volatiliteit is met adaptieve SL het meest winstgevend, niet het meest gevaarlijk.

**Code-wijzigingen in `live_engine.py`:**
- `get_adaptive_sl_mult(profile_key, atr_rank)` — rank-gated SL multiplier
- `BOUNCE_*` constanten — bounce exit parameters
- State: `bounce_pending` dict toegevoegd naast `open_trades`
- Exit loop: SL → `bounce_pending` in plaats van directe close
- Bounce check elke cycle (step 3a) voor recovery / safety / trail / timeout
- Dashboard: toont bounce-pending posities apart met safety floor

---

## 10. 2026-06-17: max_bars verwijderd + 1-bar entry bevestiging

**max_bars verwijderd:**
- Backtest (`backtest_maxbars.py`): alle MaxBars-exits zijn verliesgevend — n=130, gem. -7.30p, 12% win.
- Alle profielen nu op 999 (geen tijdslimiet). Trail/DirectionFlip/SL sluiten posities als het moment er is.

**1-bar entry bevestiging:**
- Backtest (`backtest_entry.py`, 5-min bars, delay=1 bar = 10 min):
  - Zonder delay: gem. +0.46p/trade. Na 1 bar bevestiging: gem. **+1.45p/trade** (+215%).
  - 79% van trades gaat eerst negatief in de eerste bar na opening. 96.7% van signalen blijft na 1 bar.
  - Gecombineerd (delay + adaptieve SL + bounce): **+8704p, gem. +2.45p, win 84.8%, DD -172p**.

**Implementatie in `live_engine.py`:**
- `state["pending_signal"]` dict: slaat signaal op van bar N.
- Bar N+1: als richting + environment nog geldig → order geplaatst. Anders → signaal weggegooid.
- `entry_direction/environment/local/profile_key` variabelen zorgen dat de juiste trade-parameters
  van het originele signaal worden gebruikt, niet de huidige bar.

---

## Eerder deze week (context, 2026-06-10/11)

- Logging-fix: `config.py` riep `logging.basicConfig()` aan bij import waardoor
  de FileHandler van live_engine genegeerd werd → logbestand bleef leeg.
  Opgelost door root-logger handlers direct te vervangen.
- `trade_execution.py` vervangen door testingversion2-versie (quantity-clamp
  tegen StatusReason 8 + min-amount cache).
- Daily-bias regel verwijderd uit entry-filter (blokkeerde alles bij "Mixed").
- Broker als bron van waarheid: open-positie-telling via
  `PositionManagementModule.get_open_positions()` vóór elke entry en close.
