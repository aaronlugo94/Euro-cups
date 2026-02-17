import pandas as pd
import numpy as np
import requests
import io
import difflib
import time
import schedule
import os
import csv
import json
import re
import math
import traceback
from datetime import datetime, timedelta
from collections import Counter

# --- CONFIGURACIÓN v85.2 (SURGICAL + EURO CUP INJECTION) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "02:20" 

# AJUSTES DE MODELO
SIMULATION_RUNS = 20000 
DECAY_ALPHA = 0.88          
MIN_EV_THRESHOLD = 0.02
SEASON = '2526'

# --- 🏆 SECCIÓN DE COPAS EUROPEAS (MANUAL) 🏆 ---
# Escribe aquí los partidos de Champions/Europa League que quieras analizar hoy.
# El bot buscará estos equipos en sus ligas domésticas y cruzará los datos.
# Formato: ('Local', 'Visitante')
MANUAL_MATCHES = [
    ('Galatasaray', 'Juventus'),
    ('Dortmund', 'Atalanta'),
    ('Monaco', 'PSG'),
    ('Benfica', 'Real Madrid'),
    ('Qarabag FK', 'Newcastle'),
    ('Olympiacos', 'Leverkusen'),
    ('Bodo/Glimt', 'Inter')
]

# --- 💾 PERSISTENCIA RAILWAY ---
VOLUME_PATH = "/app/data" 
if os.path.exists(VOLUME_PATH):
    HISTORY_FILE = os.path.join(VOLUME_PATH, "historial_omni_v85.csv")
    print(f"💾 USANDO VOLUMEN PERSISTENTE: {HISTORY_FILE}", flush=True)
else:
    HISTORY_FILE = "historial_omni_v85.csv"
    print("⚠️ USANDO ALMACENAMIENTO EFÍMERO", flush=True)

# GESTIÓN DE RIESGO
KELLY_FRACTION = 0.20        
MAX_STAKE_PCT = 0.04        

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

# CONFIGURACIÓN DINÁMICA DE LIGAS (Tier ajustado a float para cálculo inter-ligas)
LEAGUE_CONFIG = {
    'E0':  {'name': '🇬🇧 PREMIER', 'tier': 1.00, 'm_weight': 0.85, 'min_ev': 0.02},
    'SP1': {'name': '🇪🇸 LA LIGA', 'tier': 1.00, 'm_weight': 0.85, 'min_ev': 0.02},
    'I1':  {'name': '🇮🇹 SERIE A', 'tier': 1.00, 'm_weight': 0.82, 'min_ev': 0.02},
    'D1':  {'name': '🇩🇪 BUNDES',  'tier': 1.00, 'm_weight': 0.82, 'min_ev': 0.02},
    'F1':  {'name': '🇫🇷 LIGUE 1', 'tier': 0.90, 'm_weight': 0.80, 'min_ev': 0.025},
    'P1':  {'name': '🇵🇹 PORTUGAL','tier': 0.85, 'm_weight': 0.70, 'min_ev': 0.03},
    'N1':  {'name': '🇳🇱 HOLANDA', 'tier': 0.85, 'm_weight': 0.70, 'min_ev': 0.03},
    'B1':  {'name': '🇧🇪 BELGICA', 'tier': 0.80, 'm_weight': 0.65, 'min_ev': 0.04},
    'T1':  {'name': '🇹🇷 TURQUIA', 'tier': 0.75, 'm_weight': 0.60, 'min_ev': 0.04},
    'G1':  {'name': '🇬🇷 GRECIA',  'tier': 0.70, 'm_weight': 0.60, 'min_ev': 0.04},
    'SC0': {'name': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 ESCOCIA', 'tier': 0.70, 'm_weight': 0.60, 'min_ev': 0.04}
}

# --- DIAGNÓSTICO ---
SDK_STATUS = "UNKNOWN"
try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
    SDK_STATUS = "✅ LIBRERÍA INSTALADA."
except ImportError as ie:
    SDK_AVAILABLE = False
    SDK_STATUS = f"❌ ERROR: {ie}"

class OmniHybridBot:
    def __init__(self):
        self.history_cache = {} 
        self.daily_picks_buffer = [] 
        self.handicap_buffer = [] 
        self.global_db = {} # Nueva DB para Copas
        
        print("--- ENGINE v85.2 SURGICAL + CUPS STARTED ---", flush=True)
        self.send_msg(f"🔧 <b>INICIANDO v85.2</b>\n(Surgical Core + Cup Injection)\n📂 CSV: {HISTORY_FILE}")
        
        self._init_history_file()
        
        self.ai_client = None
        if SDK_AVAILABLE and GEMINI_API_KEY:
            try:
                self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
                print("🧠 Gemini SDK: Cliente Creado.", flush=True)
            except Exception as e:
                self.send_msg(f"⚠️ Error Cliente Gemini: {e}")

    def _init_history_file(self):
        if not os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Date', 'League', 'Home', 'Away', 'Pick', 'Market', 'Prob', 'Odd', 'EV', 'Status', 'Stake', 'Profit', 'FTHG', 'FTAG'])
            except Exception as e: print(f"Error CSV: {e}")

    def sanitize_text(self, text):
        text = text.replace("```html", "").replace("```", "")
        text = re.sub(r'<!DOCTYPE.*?>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<html.*?>|</html>|<head>.*?</head>|<body.*?>|</body>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = text.replace("**", "") 
        return text.strip()

    def send_msg(self, text, retry_count=0, use_html=True):
        if not TELEGRAM_TOKEN: return
        if use_html: text = self.sanitize_text(text)
        if len(text) > 4000:
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for chunk in chunks: self.send_msg(chunk, retry_count, use_html)
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML" if use_html else None}
        try:
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 200: return
        except Exception as e: print(f"Error Telegram: {e}", flush=True)

    def dec_to_am(self, decimal_odd):
        if decimal_odd <= 1.01: return "-10000"
        if decimal_odd >= 2.00: return f"+{int((decimal_odd - 1) * 100)}"
        else: return f"{int(-100 / (decimal_odd - 1))}"

    def call_gemini(self, prompt):
        if not SDK_AVAILABLE or not self.ai_client: return "❌ SDK no disponible."
        try:
            config = types.GenerateContentConfig(temperature=0.7)
            r = self.ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt, config=config)
            return r.text if r.text else "⚠️ Respuesta vacía."
        except Exception as e: return f"⚠️ Error Gemini: {str(e)[:100]}"

    # --- CÁLCULO CORE (NO TOCAR) ---
    def calculate_team_stats(self, df, team):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(6)
        if len(matches) < 3: return 1.0, 1.0
        w_att = 0; w_def = 0; total_w = 0
        for i, (_, row) in enumerate(matches.iterrows()):
            weight = pow(DECAY_ALPHA, 5 - i); total_w += weight
            if row['HomeTeam'] == team:
                att = (row['FTHG'] * 0.6) + ((row.get('HST', row['FTHG']*3)/3) * 0.4)
                def_weak = (row['FTAG'] * 0.6) + ((row.get('AST', row['FTAG']*3)/3) * 0.4)
            else:
                att = (row['FTAG'] * 0.6) + ((row.get('AST', row['FTAG']*3)/3) * 0.4)
                def_weak = (row['FTHG'] * 0.6) + ((row.get('HST', row['FTHG']*3)/3) * 0.4)
            w_att += att * weight; w_def += def_weak * weight
        return w_att / total_w, w_def / total_w

    def get_league_data(self, div):
        url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{div}.csv"
        try:
            r = requests.get(url, headers={'User-Agent': USER_AGENTS[0]}, timeout=15)
            if r.status_code != 200: return None
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
            df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
            matches_played = df.dropna(subset=['FTHG', 'FTAG'])
            if len(matches_played) > 0: avg_g = matches_played.FTHG.mean() + matches_played.FTAG.mean()
            else: avg_g = 2.5
            teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
            team_stats = {}
            avg_a = 0; avg_d = 0; cnt = 0
            for t in teams:
                a, d = self.calculate_team_stats(matches_played, t)
                team_stats[t] = {'att': a, 'def': d}
                avg_a += a; avg_d += d; cnt += 1
            if cnt > 0: avg_a /= cnt; avg_d /= cnt
            else: avg_a = 1; avg_d = 1
            
            # Guardamos en DB Global para las copas
            tier = LEAGUE_CONFIG.get(div, {}).get('tier', 1.0)
            for t, s in team_stats.items():
                self.global_db[t] = {
                    'att': s['att']/avg_a, 
                    'def': s['def']/avg_d, 
                    'tier': tier, 
                    'avg_g': avg_g, 
                    'raw_df': df,
                    'league_name': LEAGUE_CONFIG.get(div, {}).get('name', div)
                }

            norm_stats = {t: {'att': s['att']/avg_a, 'def': s['def']/avg_d} for t, s in team_stats.items()}
            return {'stats': norm_stats, 'teams': teams, 'raw_df': df, 'avg_g': avg_g}
        except: return None

    # --- MOTOR MATEMÁTICO ---
    def poisson_prob(self, k, lamb):
        return (math.pow(lamb, k) * math.exp(-lamb)) / math.factorial(k)

    def calculate_dixon_coles_1x2(self, lambda_h, lambda_a):
        rho = -0.13; prob_h, prob_d, prob_a = 0.0, 0.0, 0.0
        for x in range(7):
            for y in range(7):
                p = self.poisson_prob(x, lambda_h) * self.poisson_prob(y, lambda_a)
                correction = 1.0
                if x==0 and y==0: correction = 1 - (lambda_h * lambda_a * rho)
                elif x==0 and y==1: correction = 1 + (lambda_h * rho)
                elif x==1 and y==0: correction = 1 + (lambda_a * rho)
                elif x==1 and y==1: correction = 1 - (rho)
                final_p = p * correction
                if x > y: prob_h += final_p
                elif x == y: prob_d += final_p
                else: prob_a += final_p
        return prob_h, prob_d, prob_a

    def calibrate_goal_prob(self, p):
        return 0.5 + (p - 0.5) * 0.75

    def simulate_match(self, home, away, league_data, market_odds, m_weight_config):
        h_st = league_data['stats'].get(home, {'att':1.0, 'def':1.0})
        a_st = league_data['stats'].get(away, {'att':1.0, 'def':1.0})
        avg_g = league_data['avg_g'] / 2
        
        # --- LÓGICA INTER-LIGA (SOLO SI SE LLAMA DESDE MANUAL MATCHES) ---
        # Si venimos de copas, league_data['inter_league'] será True
        if league_data.get('inter_league', False):
            # Ajuste de Tier: Si Home Tier > Away Tier, Home Att sube, Away Def baja
            h_tier = league_data['h_tier']; a_tier = league_data['a_tier']
            tier_diff = h_tier - a_tier
            
            # Factores de ajuste quirúrgicos
            h_st['att'] *= (1 + tier_diff * 0.40)
            h_st['def'] *= (1 - tier_diff * 0.20)
            a_st['att'] *= (1 - tier_diff * 0.40)
            a_st['def'] *= (1 + tier_diff * 0.20)
            
            # Ventaja de localía extra en Europa
            avg_g = (league_data['h_avg_g'] + league_data['a_avg_g']) / 4 # Promedio de promedios
            lambda_h = h_st['att'] * a_st['def'] * avg_g * 1.15 
            lambda_a = a_st['att'] * h_st['def'] * avg_g
        else:
            # Lógica Doméstica Estándar (NO TOCADA)
            lambda_h = h_st['att'] * a_st['def'] * avg_g * 1.10
            lambda_a = a_st['att'] * h_st['def'] * avg_g
        
        h_sim = np.random.poisson(lambda_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(lambda_a, SIMULATION_RUNS)
        
        prob_h, prob_d, prob_a = self.calculate_dixon_coles_1x2(lambda_h, lambda_a)
        
        if market_odds['H'] > 0:
            margin = 1.05 
            implied_h = (1 / market_odds['H']) / margin
            implied_a = (1 / market_odds['A']) / margin
            implied_d = (1 / market_odds['D']) / margin
            
            w_market = m_weight_config
            w_model = 1.0 - w_market
            
            raw_h = (implied_h * w_market) + (prob_h * w_model)
            raw_a = (implied_a * w_market) + (prob_a * w_model)
            raw_d = (implied_d * w_market) + (prob_d * w_model)
            total = raw_h + raw_a + raw_d
            prob_h, prob_a, prob_d = raw_h/total, raw_a/total, raw_d/total

        over25_raw = np.mean((h_sim + a_sim) > 2.5)
        over25 = self.calibrate_goal_prob(over25_raw)
        if (lambda_h + lambda_a) > 2.6 and abs(lambda_h - lambda_a) > 1.4: over25 *= 0.88 
        
        implied_over = 0.5
        if market_odds.get('O25', 0) > 1:
            implied_over = (1 / market_odds['O25']) / 1.05
            if abs(over25 - implied_over) > 0.08:
                over25 = (over25 * 0.75) + (implied_over * 0.25)
            
        btts = np.mean((h_sim > 0) & (a_sim > 0))
        
        xg_sum = lambda_h + lambda_a
        xg_diff = abs(lambda_h - lambda_a)
        xg_score = min(1, max(0, (xg_sum - 1.8) / 1.8))
        balance = max(0, 1 - (xg_diff / xg_sum)) if xg_sum > 0 else 0
        
        extreme_bonus = max(0, 0.65 - abs(over25 - 0.5)) 
        gcs = (0.30 * xg_score + 0.20 * balance + 0.20 * extreme_bonus * 2 + 0.15 * (1-abs(btts-over25))) * 100

        h_sim_cap = np.minimum(h_sim, 6)
        a_sim_cap = np.minimum(a_sim, 6)
        most_common, count = Counter(zip(h_sim_cap, a_sim_cap)).most_common(1)[0]
        cs_str = f"{most_common[0]}-{most_common[1]}"
        cs_prob = (count / SIMULATION_RUNS) * 100
        
        ah_h_minus = np.mean((h_sim - 1.5) > a_sim); ah_a_minus = np.mean((a_sim - 1.5) > h_sim)
        ah_h_plus = np.mean((h_sim + 1.5) > a_sim); ah_a_plus = np.mean((a_sim + 1.5) > h_sim)
        den_dnb = prob_h + prob_a
        dnb_h = prob_h / den_dnb if den_dnb > 0 else 0.5
        dnb_a = prob_a / den_dnb if den_dnb > 0 else 0.5

        return {
            'lambdas': (lambda_h, lambda_a), 'stats': (h_st, a_st),
            '1x2': (prob_h, prob_d, prob_a), 'goals': (over25, btts),
            'dc': (prob_h+prob_d, prob_a+prob_d), 'dnb': (dnb_h, dnb_a),
            'ah': (ah_h_minus, ah_a_minus, ah_h_plus, ah_a_plus),
            'gcs': gcs, 'cs': (cs_str, cs_prob), 'm_weight': m_weight_config
        }

    def get_avg_odds(self, row):
        def get_avg(cols):
            vals = [float(row[c]) for c in cols if row.get(c) and str(row[c]).replace('.','').isdigit()]
            return sum(vals)/len(vals) if vals else 0.0
        return {
            'H': get_avg(['B365H', 'PSH', 'WHH']), 'D': get_avg(['B365D', 'PSD', 'WHD']),
            'A': get_avg(['B365A', 'PSA', 'WHA']), 'O25': get_avg(['B365>2.5', 'P>2.5', 'WH>2.5']),
            'BTTS_Y': get_avg(['BbAvBBTS', 'B365BTTSY'])
        }

    def find_best_value(self, sim, odds, min_ev_league):
        candidates = []
        handicap_candidates = []
        
        def add(name, market, prob, odd, gcs=None):
            if odd < 1.05: return
            ev = (prob * odd) - 1
            status = "VALID"; reason = "OK"
            
            if ev < min_ev_league: status="REJECTED"; reason=f"EV Bajo ({ev*100:.1f}%)"
            elif prob < 0.35: status="REJECTED"; reason=f"Riesgo ({prob*100:.0f}%)"
            elif ev > 0.45: status="REJECTED"; reason="Error Modelo"
            if market == 'GOALS':
                if gcs < 55: status="REJECTED"; reason=f"GCS Pobre ({gcs:.0f})"
                elif prob > 0.65 or prob < 0.35: status="REJECTED"; reason="Prob Extrema"
            
            if market == "HANDI" and odd < 1.60:
                status = "BACKUP"; reason = "Reserva Parlay"
            
            base_score = ev * (prob ** 1.5)
            if 1.70 <= odd <= 2.50: base_score *= 1.3 
            
            item = {'pick': name, 'market': market, 'prob': prob, 'odd': odd, 'ev': ev, 'score': base_score, 'status': status, 'reason': reason, 'gcs': gcs}
            
            if market == "HANDI": handicap_candidates.append(item)
            else: candidates.append(item)

        if odds['H'] > 0:
            add("GANA HOME", "1X2", sim['1x2'][0], odds['H'])
            add("GANA AWAY", "1X2", sim['1x2'][2], odds['A'])
            add("DNB HOME", "DNB", sim['dnb'][0], (odds['H'] * (1 - (1/odds['D']))) * 0.94)
            add("DNB AWAY", "DNB", sim['dnb'][1], (odds['A'] * (1 - (1/odds['D']))) * 0.94)
            add("DC 1X", "Double Chance", sim['dc'][0], 1 / ((1/odds['H']) + (1/odds['D'])) * 0.94)
            add("DC X2", "Double Chance", sim['dc'][1], 1 / ((1/odds['A']) + (1/odds['D'])) * 0.94)

        if odds['O25'] > 0:
            add("OVER 2.5 GOLES", "GOALS", sim['goals'][0], odds['O25'], sim['gcs'])
            add("UNDER 2.5 GOLES", "GOALS", 1-sim['goals'][0], 1 / (1 - (1/odds['O25'] * 1.05)), sim['gcs'])
        if odds['BTTS_Y'] > 0:
            add("BTTS SÍ", "BTTS", sim['goals'][1], odds['BTTS_Y'])
            add("BTTS NO", "BTTS", 1-sim['goals'][1], 1 / (1 - (1/odds['BTTS_Y']*1.05)))
        
        ah_h_plus = sim['ah'][2]; ah_a_plus = sim['ah'][3]
        if ah_h_plus > 0.90: add("HANDICAP H +1.5", "HANDI", ah_h_plus, 1.15)
        if ah_a_plus > 0.90: add("HANDICAP A +1.5", "HANDI", ah_a_plus, 1.15)

        best_handi = None
        if handicap_candidates:
            handicap_candidates.sort(key=lambda x: x['ev'], reverse=True)
            best_handi = handicap_candidates[0]

        if not candidates: return None, best_handi
        
        principales = [c for c in candidates if c['status'] == "VALID"]
        if principales:
            principales.sort(key=lambda x: x['score'], reverse=True)
            return principales[0], best_handi
        
        candidates.sort(key=lambda x: x['ev'], reverse=True)
        return candidates[0], best_handi

    def get_kelly_stake(self, prob, odds, market, gcs=None):
        if odds <= 1.0: return 0.0
        q = 1 - prob; b = odds - 1
        full = (b * prob - q) / b
        stake = full * KELLY_FRACTION
        
        if market in ['GOALS', 'BTTS'] and gcs:
            confidence_factor = min(1.0, gcs / 75)
            stake *= confidence_factor
        
        return max(0.0, min(stake, MAX_STAKE_PCT))

    def get_team_form_icon(self, df, team):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(5)
        if len(matches) == 0: return "➡️"
        points = 0; possible = len(matches) * 3
        for _, row in matches.iterrows():
            if row['HomeTeam'] == team:
                if row['FTHG'] > row['FTAG']: points += 3
                elif row['FTHG'] == row['FTAG']: points += 1
            else:
                if row['FTAG'] > row['FTHG']: points += 3
                elif row['FTAG'] == row['FTHG']: points += 1
        pct = points / possible
        if pct >= 0.7: return "🔥"; 
        if pct <= 0.3: return "🧊"; 
        return "➡️"

    # --- AUDITORÍA & PNL ---
    def check_bet_result(self, pick, market, fthg, ftag):
        if math.isnan(fthg) or math.isnan(ftag): return "PENDING"
        hg = int(fthg); ag = int(ftag)
        win = False
        
        if market == "1X2":
            if "HOME" in pick and hg > ag: win=True
            elif "AWAY" in pick and ag > hg: win=True
            elif "DRAW" in pick and hg == ag: win=True
        elif market == "DNB":
            if hg == ag: return "PUSH"
            if "HOME" in pick and hg > ag: win=True
            elif "AWAY" in pick and ag > hg: win=True
        elif market == "Double Chance":
            if ("1X" in pick and hg >= ag) or ("X2" in pick and ag >= hg): win=True
        elif market == "GOALS":
            if "OVER" in pick and (hg+ag) > 2.5: win=True
            elif "UNDER" in pick and (hg+ag) < 2.5: win=True
        elif market == "BTTS":
            if "SÍ" in pick and (hg>0 and ag>0): win=True
            elif "NO" in pick and not (hg>0 and ag>0): win=True
        return "WIN" if win else "LOSS"

    def run_audit(self):
        if not os.path.exists(HISTORY_FILE): return
        league_data_map = {}
        for div in LEAGUE_CONFIG.keys(): league_data_map[div] = self.get_league_data(div)
        rows = []; audit_buffer = []
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f); header = next(reader); rows.append(header)
                for row in reader:
                    status = row[9]
                    if status in ['VALID', '0', 'BACKUP'] and row[1] in league_data_map:
                        div = row[1]; home = row[2]; away = row[3]; pick = row[4]; market = row[5]
                        odd = float(row[7]); stake = float(row[10]) if row[10] else 0.0
                        data = league_data_map.get(div)
                        if data and not data['raw_df'].empty:
                            raw = data['raw_df']
                            match = raw[(raw['HomeTeam'] == home) & (raw['AwayTeam'] == away)]
                            if not match.empty:
                                fthg = match.iloc[0]['FTHG']; ftag = match.iloc[0]['FTAG']
                                res = self.check_bet_result(pick, market, fthg, ftag)
                                if res in ["WIN", "LOSS", "PUSH"]:
                                    row[9] = res; row[12] = fthg; row[13] = ftag
                                    if res == "WIN": profit = (stake * odd) - stake
                                    elif res == "LOSS": profit = -stake
                                    else: profit = 0.0
                                    row[11] = round(profit, 2)
                                    if stake > 0: audit_buffer.append(f"{pick}: {res} ({row[11]} U)")
                    rows.append(row)
            with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f); writer.writerows(rows)
            if audit_buffer:
                audit_txt = "\n".join(audit_buffer)
                try: self.send_msg(f"🔬 <b>AUDITORÍA DIARIA:</b>\n{audit_txt}")
                except: pass
        except Exception as e: print(f"Audit Error: {e}")

    def calculate_pnl(self):
        if not os.path.exists(HISTORY_FILE): return
        try:
            df = pd.read_csv(HISTORY_FILE)
            df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
            df['Stake'] = pd.to_numeric(df['Stake'], errors='coerce').fillna(0)
            total_profit = df['Profit'].sum()
            total_invested = df['Stake'].sum()
            roi = (total_profit / total_invested * 100) if total_invested > 0 else 0.0
            self.send_msg(f"💰 <b>PnL TOTAL:</b> {total_profit:+.2f} U | ROI: {roi:.1f}%")
        except: pass

    def generate_final_summary(self):
        if not self.daily_picks_buffer and not self.handicap_buffer: return
        self.send_msg("⏳ <b>El Jefe de Estrategia está diseñando las jugadas maestras...</b>")
        picks_text = "\n".join(self.daily_picks_buffer)
        handi_text = "\n".join(self.handicap_buffer)
        prompt = f"""
        Actúa como Jefe de Estrategia de Apuestas.
        PICKS OFICIALES (Validados):
        {picks_text}
        PICKS SEGUROS (Handicaps, usar solo para Parlay):
        {handi_text}
        Genera un reporte breve con:
        1. 💎 LA JOYA: (El mejor pick oficial).
        2. 🛡️ EL BANKER: (El pick más seguro).
        3. 🎲 PARLAY SEGURO: (2 picks seguros).
        4. 🚀 PARLAY DE VALOR: (2 joyas).
        USA SOLO negritas <b> y saltos de linea. NO uses Markdown (**).
        """
        try:
            ai_resp = self.call_gemini(prompt)
            self.send_msg(ai_resp)
        except Exception as e: self.send_msg(f"⚠️ Error Gemini: {e}")

    # --- FUNCIÓN DE BÚSQUEDA PARA COPAS ---
    def find_team_in_global(self, team_name):
        # 1. Búsqueda exacta
        if team_name in self.global_db: return self.global_db[team_name], team_name
        # 2. Búsqueda aproximada
        matches = difflib.get_close_matches(team_name, self.global_db.keys(), n=1, cutoff=0.6)
        if matches: return self.global_db[matches[0]], matches[0]
        return None, None

    def run_analysis(self):
        self.run_audit()
        self.calculate_pnl()
        
        self.daily_picks_buffer = [] 
        self.handicap_buffer = []
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando v85.2 SURGICAL: {today}", flush=True)
        
        # 1. Cargar todas las ligas para poblar la Global DB (necesario para Copas)
        print("🌍 Cargando datos de ligas para Copas...", flush=True)
        for div in LEAGUE_CONFIG:
            self.get_league_data(div) # Esto rellena self.global_db
        
        # 2. Análisis Ligas Domésticas (Fixture Automático)
        ts = int(time.time())
        url_fixt = f"https://www.football-data.co.uk/fixtures.csv?t={ts}"
        try:
            r = requests.get(url_fixt, headers={'User-Agent': USER_AGENTS[0]}, timeout=20)
            if r.status_code==200:
                try: content = r.content.decode('utf-8-sig')
                except: content = r.content.decode('latin-1')
                df = pd.read_csv(io.StringIO(content), on_bad_lines='skip')
                df.columns = df.columns.str.strip().str.replace('ï»¿', '')
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                
                target_date = pd.to_datetime(today, dayfirst=True)
                daily = df[(df['Date'] >= target_date) & (df['Date'] <= target_date + timedelta(days=1))]
                
                self.send_msg(f"🔎 <b>Analizando {len(daily)} partidos domésticos...</b>")
                
                for idx, row in daily.iterrows():
                    div = row.get('Div')
                    if div not in LEAGUE_CONFIG: continue
                    data = self.get_league_data(div)
                    if not data: continue
                    
                    rh = difflib.get_close_matches(row['HomeTeam'], data['teams'], n=1, cutoff=0.6)
                    ra = difflib.get_close_matches(row['AwayTeam'], data['teams'], n=1, cutoff=0.6)
                    if not rh or not ra: continue
                    rh = rh[0]; ra = ra[0]
                    
                    m_odds = self.get_avg_odds(row)
                    m_weight = LEAGUE_CONFIG[div].get('m_weight', 0.70)
                    sim = self.simulate_match(rh, ra, data, m_odds, m_weight)
                    
                    min_ev = LEAGUE_CONFIG[div].get('min_ev', 0.02)
                    best_bet, best_handi = self.find_best_value(sim, m_odds, min_ev)
                    self.process_match_output(div, rh, ra, data, sim, best_bet, best_handi, today)
        except: pass

        # 3. Análisis Manual Copas Europeas (Inyección)
        if MANUAL_MATCHES:
            self.send_msg(f"🏆 <b>ANALIZANDO {len(MANUAL_MATCHES)} PARTIDOS DE COPA</b>")
            for home_input, away_input in MANUAL_MATCHES:
                h_data, real_h = self.find_team_in_global(home_input)
                a_data, real_a = self.find_team_in_global(away_input)
                
                if h_data and a_data:
                    # Construir estructura de datos híbrida para simulación
                    hybrid_data = {
                        'stats': {real_h: {'att': h_data['att'], 'def': h_data['def']}, 
                                  real_a: {'att': a_data['att'], 'def': a_data['def']}},
                        'avg_g': (h_data['avg_g'] + a_data['avg_g']) / 2,
                        'inter_league': True,
                        'h_tier': h_data['tier'], 'a_tier': a_data['tier'],
                        'h_avg_g': h_data['avg_g'], 'a_avg_g': a_data['avg_g']
                    }
                    
                    # Cuotas dummy (o 0 si no tenemos, el sistema calculará probabilidad pura)
                    dummy_odds = {'H':0, 'D':0, 'A':0, 'O25':0, 'BTTS_Y':0}
                    
                    # Simular
                    sim = self.simulate_match(real_h, real_a, hybrid_data, dummy_odds, 0.5)
                    
                    # Encontrar valor (usando cuotas 0 para ver solo prob pura)
                    # Forzamos probabilidad como si la cuota fuera 2.0 para ver si pasa filtro de calidad
                    fake_ev_odds = {'H': 2.0, 'D': 3.0, 'A': 2.0, 'O25': 1.9, 'BTTS_Y': 1.9} 
                    
                    # Usamos logic manual para output de copas
                    self.process_cup_output("🇪🇺 EUROPA", real_h, real_a, h_data, sim)

        if len(self.daily_picks_buffer) > 0 or len(self.handicap_buffer) > 0:
            self.generate_final_summary()
        else:
            self.send_msg("🧹 Barrido completado.")

    def process_cup_output(self, league_name, rh, ra, data, sim):
        # Versión simplificada del output para Copas (ya que no hay cuotas reales)
        ph, pd_raw, pa = sim['1x2']
        btts = sim['goals'][1]; ov25 = sim['goals'][0]
        lambdas = sim['lambdas']; cs_str, cs_prob = sim['cs']
        
        # Pick lógico basado en probabilidad pura
        pick = "SKIP"
        if ph > 0.45: pick = f"GANA {rh}"
        elif pa > 0.40: pick = f"GANA {ra}"
        elif ov25 > 0.55: pick = "OVER 2.5 GOLES"
        
        # Usamos el mismo formato visual
        form_h = self.get_team_form_icon(data['raw_df'], rh) # Usamos el DF de su liga original
        # Para visitante necesitamos su DF original, que está en global_db pero es complejo extraer aquí rápido
        # Simplificamos icono visitante
        form_a = "🛡️" 
        
        msg = (
            f"🛡️ <b>ANÁLISIS COPA</b> | {league_name}\n"
            f"⚽ <b>{rh}</b> {form_h} vs {form_a} <b>{ra}</b>\n"
            f"───────────────\n"
            f"🎯 PROYECCIÓN: <b>{pick}</b>\n"
            f"🧠 Prob Gana: <b>{ph*100:.1f}%</b> vs {pa*100:.1f}%\n"
            f"───────────────\n"
            f"📊 <b>X-RAY:</b>\n"
            f"• 1X2: {ph*100:.0f}% | {pd_raw*100:.0f}% | {pa*100:.0f}%\n"
            f"• Goals: Over {ov25*100:.0f}% | BTTS {btts*100:.0f}%\n"
            f"───────────────\n"
            f"🎯 Marcador: <b>{cs_str}</b> ({cs_prob:.1f}%)\n"
            f"⚔️ PODER (xG):\n"
            f"🏠 {rh}: <b>{lambdas[0]:.2f}</b> gls\n"
            f"✈️ {ra}: <b>{lambdas[1]:.2f}</b> gls"
        )
        self.send_msg(msg)
        self.daily_picks_buffer.append(f"🏆 {rh} vs {ra}: {pick} (Prob: {ph if 'HOME' in pick else pa:.2f})")

    def process_match_output(self, div, rh, ra, data, sim, best_bet, best_handi, today):
        if best_bet:
            is_valid = best_bet['status'] == "VALID"
            if is_valid:
                icon = "🎯"; status_line = "✅ <b>PICK ACTIVO</b>"
                gcs_val = best_bet.get('gcs', 0)
                stake = self.get_kelly_stake(best_bet['prob'], best_bet['odd'], best_bet['market'], gcs_val)
                stake_txt = f"{stake*100:.2f}%"
                tag = "[VALID]"
                self.daily_picks_buffer.append(f"{tag} {rh} vs {ra}: {best_bet['pick']} @ {best_bet['odd']:.2f} (EV: {best_bet['ev']*100:.1f}%)")
            else:
                icon = "🚫"; status_line = f"🚫 <b>NO BET</b> ({best_bet['reason']})"
                stake = 0.0; stake_txt = "Skipped"
                icon_pick = "⚠️"

            if best_handi:
                self.handicap_buffer.append(f"{rh} vs {ra}: {best_handi['pick']} @ {best_handi['odd']:.2f}")

            form_h = self.get_team_form_icon(data['raw_df'], rh)
            form_a = self.get_team_form_icon(data['raw_df'], ra)
            ph, pd_raw, pa = sim['1x2']; dc1x, dcx2 = sim['dc']; dnb_h, dnb_a = sim['dnb']
            btts = sim['goals'][1]; ov25 = sim['goals'][0]; ah_h_m15, ah_a_m15, ah_h_p15, ah_a_p15 = sim['ah']
            h_stats, a_stats = sim['stats']; lambdas = sim['lambdas']; cs_str, cs_prob = sim['cs']
            fair_odd_us = self.dec_to_am(1/best_bet['prob']) if best_bet['prob'] > 0 else "-"
            
            pick_icon_display = "🎯" if is_valid else "⚠️"
            
            msg = (
                f"🛡️ <b>ANÁLISIS v85</b> | {LEAGUE_CONFIG[div]['name']}\n"
                f"⚽ <b>{rh}</b> {form_h} vs {form_a} <b>{ra}</b>\n"
                f"───────────────\n"
                f"{status_line}\n"
                f"{pick_icon_display} PICK: <b>{best_bet['pick']}</b> ({best_bet['market']})\n"
                f"⚖️ Cuota Avg: <b>{self.dec_to_am(best_bet['odd'])}</b> ({best_bet['odd']:.2f})\n"
                f"🧠 Prob: <b>{best_bet['prob']*100:.1f}%</b> (Fair: {fair_odd_us})\n"
                f"📈 EV: <b>+{best_bet['ev']*100:.1f}%</b>\n"
                f"🏦 Stake: {stake_txt}\n"
                f"───────────────\n"
                f"📊 <b>X-RAY (Probabilidades):</b>\n"
                f"• 1X2: {ph*100:.0f}% | {pd_raw*100:.0f}% | {pa*100:.0f}%\n"
                f"• DC: 1X {dc1x*100:.0f}% | X2 {dcx2*100:.0f}%\n"
                f"• DNB: H {dnb_h*100:.0f}% | A {dnb_a*100:.0f}%\n"
                f"• BTTS: Sí {btts*100:.0f}% | No {(1-btts)*100:.0f}%\n"
                f"• Goals: Over {ov25*100:.0f}% | Under {(1-ov25)*100:.0f}%\n"
                f"• Handi -1.5: H {ah_h_m15*100:.0f}% | A {ah_a_m15*100:.0f}%\n"
                f"• Handi +1.5: H {ah_h_p15*100:.0f}% | A {ah_a_p15*100:.0f}%\n"
                f"───────────────\n"
                f"🎯 Marcador Probable: <b>{cs_str}</b> ({cs_prob:.1f}%)\n"
                f"⚔️ PODER (Att / Def / Exp.Goals):\n"
                f"🏠 {rh}: {h_stats['att']:.2f} / {h_stats['def']:.2f} => <b>{lambdas[0]:.2f}</b> gls\n"
                f"✈️ {ra}: {a_stats['att']:.2f} / {a_stats['def']:.2f} => <b>{lambdas[1]:.2f}</b> gls\n"
                f"⚖️ Confianza en Mercado: {sim['m_weight']*100:.0f}%"
            )
            self.send_msg(msg)
            
            with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([today, div, rh, ra, best_bet['pick'], best_bet['market'], best_bet['prob'], best_bet['odd'], best_bet['ev'], best_bet['status'], stake, 0, "", ""])

if __name__ == "__main__":
    bot = OmniHybridBot()
    if os.getenv("SELF_TEST", "False") == "True": 
        bot.run_analysis()
    
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: 
        schedule.run_pending()
        time.sleep(60)
