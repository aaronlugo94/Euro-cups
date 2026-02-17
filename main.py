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

# --- CONFIGURACIÓN v87.1 (DATA-DRIVEN ANALYST) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "03:24" 

# AJUSTES DE MODELO
SIMULATION_RUNS = 20000 
DECAY_ALPHA = 0.88          
SEASON = '2526'

# --- 🏆 MANUAL MATCHES (CHAMPIONS/EUROPA) 🏆 ---
MANUAL_MATCHES = [
    ('Galatasaray', 'Juventus'),
    ('Dortmund', 'Atalanta'),
    ('Monaco', 'PSG'),
    ('Benfica', 'Real Madrid'),
    ('Qarabag FK', 'Newcastle'),
    ('Olympiacos', 'Leverkusen'),
    ('Bodo/Glimt', 'Inter'),
    ('Club Brugge', 'Atletico Madrid')
]

# --- 💾 PERSISTENCIA ---
VOLUME_PATH = "/app/data" 
if os.path.exists(VOLUME_PATH):
    HISTORY_FILE = os.path.join(VOLUME_PATH, "historial_omni_v87.csv")
else:
    HISTORY_FILE = "historial_omni_v87.csv"

# GESTIÓN DE RIESGO
MAX_STAKE_PCT = 0.03 

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

# FILTRO VIP
TOP_5_LEAGUES = ['E0', 'SP1', 'I1', 'D1', 'F1']

# CONFIGURACIÓN DE LIGAS
LEAGUE_CONFIG = {
    'E0':  {'name': '🇬🇧 PREMIER', 'tier': 1.00, 'm_weight': 0.85},
    'SP1': {'name': '🇪🇸 LA LIGA', 'tier': 1.00, 'm_weight': 0.85},
    'I1':  {'name': '🇮🇹 SERIE A', 'tier': 1.00, 'm_weight': 0.82},
    'D1':  {'name': '🇩🇪 BUNDES',  'tier': 1.00, 'm_weight': 0.82},
    'F1':  {'name': '🇫🇷 LIGUE 1', 'tier': 0.90, 'm_weight': 0.80},
    'P1':  {'name': '🇵🇹 PORTUGAL','tier': 0.85, 'm_weight': 0.70},
    'N1':  {'name': '🇳🇱 HOLANDA', 'tier': 0.85, 'm_weight': 0.70},
    'B1':  {'name': '🇧🇪 BELGICA', 'tier': 0.80, 'm_weight': 0.65},
    'T1':  {'name': '🇹🇷 TURQUIA', 'tier': 0.75, 'm_weight': 0.60},
    'G1':  {'name': '🇬🇷 GRECIA',  'tier': 0.70, 'm_weight': 0.60},
    'SC0': {'name': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 ESCOCIA', 'tier': 0.70, 'm_weight': 0.60},
    'EU_CUP': {'name': '🏆 COPA EUROPA', 'tier': 1.00, 'm_weight': 0.50} 
}

# --- DIAGNÓSTICO ---
SDK_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
except ImportError: pass

class OmniHybridBot:
    def __init__(self):
        self.daily_picks_buffer = [] 
        self.full_reports_buffer = [] # Nuevo buffer para reportes completos (X-RAY/Poder)
        self.handicap_buffer = [] 
        self.global_db = {} 
        
        print("--- ENGINE v87.1 DATA-DRIVEN STARTED ---", flush=True)
        self.send_msg(f"🔧 <b>INICIANDO v87.1</b>\n(Gemini Analyst Mode Activado)\n📂 CSV: {HISTORY_FILE}")
        self._init_history_file()
        
        self.ai_client = None
        if SDK_AVAILABLE and GEMINI_API_KEY:
            try: self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
            except: pass

    def _init_history_file(self):
        if not os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(['Date', 'League', 'Home', 'Away', 'Pick', 'Market', 'Prob', 'Odd', 'EV', 'Status', 'Stake', 'Profit', 'FTHG', 'FTAG'])
            except: pass

    def sanitize_text(self, text):
        text = text.replace("```html", "").replace("```", "")
        text = re.sub(r'<!DOCTYPE.*?>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = text.replace("**", "") 
        return text.strip()

    def send_msg(self, text, retry_count=0, use_html=True):
        if not TELEGRAM_TOKEN: return
        if use_html: text = self.sanitize_text(text)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML" if use_html else None}
        try: requests.post(url, json=payload, timeout=20)
        except: pass

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
        except: return "⚠️ Error Gemini"

    # --- CÁLCULO CORE ---
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
            
            tier = LEAGUE_CONFIG.get(div, {}).get('tier', 1.0)
            for t, s in team_stats.items():
                self.global_db[t] = {
                    'att': s['att']/avg_a, 'def': s['def']/avg_d, 
                    'tier': tier, 'avg_g': avg_g, 'raw_df': df
                }
            norm_stats = {t: {'att': s['att']/avg_a, 'def': s['def']/avg_d} for t, s in team_stats.items()}
            return {'stats': norm_stats, 'teams': teams, 'raw_df': df, 'avg_g': avg_g}
        except: return None

    # --- SIMULADOR ---
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
        h_st = league_data['stats'].get(home, {'att':1.0, 'def':1.0}).copy()
        a_st = league_data['stats'].get(away, {'att':1.0, 'def':1.0}).copy()
        avg_g = league_data['avg_g'] / 2
        
        if league_data.get('inter_league', False):
            h_tier = league_data['h_tier']; a_tier = league_data['a_tier']
            tier_diff = h_tier - a_tier
            h_st['att'] *= (1 + tier_diff * 0.40)
            h_st['def'] *= (1 - tier_diff * 0.20)
            a_st['att'] *= (1 - tier_diff * 0.40)
            a_st['def'] *= (1 + tier_diff * 0.20)
            lambda_h = h_st['att'] * a_st['def'] * avg_g * 1.15
            lambda_a = a_st['att'] * h_st['def'] * avg_g
        else:
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
            if total > 0: prob_h, prob_a, prob_d = raw_h/total, raw_a/total, raw_d/total

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

    # --- LÓGICA WIN-FIRST ---
    def find_best_value(self, sim, odds, min_ev_league):
        candidates = []
        
        def add(name, market, prob, odd, gcs=None):
            if odd < 1.50 or odd > 2.15: return 
            if prob < 0.60: return
            ev = (prob * odd) - 1
            
            if market == "1X2" and prob < 0.65: return
            if market == "GOALS" and (not gcs or gcs < 58): return
            if ev < -0.02 or ev > 0.25: return

            score = (prob * 100)
            if market == "DNB": score += 6
            if market == "Double Chance": score += 5
            if prob >= 0.68: score += 4
            if ev > 0.03: score += min(ev * 30, 3)
            
            item = {'pick': name, 'market': market, 'prob': prob, 'odd': odd, 'ev': ev, 'score': score, 'status': "VALID", 'reason': "WIN-FIRST", 'gcs': gcs}
            candidates.append(item)

        if odds['H'] > 0:
            add("DNB HOME", "DNB", sim['dnb'][0], (odds['H'] * (1 - (1/odds['D']))) * 0.95)
            add("DNB AWAY", "DNB", sim['dnb'][1], (odds['A'] * (1 - (1/odds['D']))) * 0.95)
            add("DC 1X", "Double Chance", sim['dc'][0], 1 / ((1/odds['H']) + (1/odds['D'])) * 0.95)
            add("DC X2", "Double Chance", sim['dc'][1], 1 / ((1/odds['A']) + (1/odds['D'])) * 0.95)
            add("GANA HOME", "1X2", sim['1x2'][0], odds['H'])
            add("GANA AWAY", "1X2", sim['1x2'][2], odds['A'])
        if odds['O25'] > 0:
            add("OVER 2.5 GOLES", "GOALS", sim['goals'][0], odds['O25'], sim['gcs'])
            add("UNDER 2.5 GOLES", "GOALS", 1-sim['goals'][0], 1 / (1 - (1/odds['O25'] * 1.05)), sim['gcs'])
        
        best_handi = None
        ah_h_plus = sim['ah'][2]; ah_a_plus = sim['ah'][3]
        if ah_h_plus > 0.85: best_handi = {'pick': "HANDICAP H +1.5", 'odd': 1.15}
        elif ah_a_plus > 0.85: best_handi = {'pick': "HANDICAP A +1.5", 'odd': 1.15}

        if not candidates: return None, best_handi
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0], best_handi

    # --- STAKE ESCALONADO ---
    def get_stake(self, prob, odds, market, gcs=None):
        base = 0.01 
        if prob >= 0.65: base = 0.0125
        if prob >= 0.70: base = 0.015
        if market in ["DNB", "Double Chance"]: base *= 1.1
        if market == 'GOALS': base *= 0.9
        return min(base, 0.02)

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
        if pct >= 0.7: return "🔥"
        if pct <= 0.3: return "🧊"
        return "➡️"

    # --- PNL & AUDITORÍA ---
    def check_bet_result(self, pick, market, fthg, ftag):
        if math.isnan(fthg): return "PENDING"
        hg = int(fthg); ag = int(ftag); win = False
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
            roi = (total_profit / df['Stake'].sum() * 100) if df['Stake'].sum() > 0 else 0
            self.send_msg(f"💰 <b>PnL TOTAL:</b> {total_profit:+.2f} U | ROI: {roi:.1f}%")
        except: pass

    # --- GEMINI ANALYST ---
    def generate_final_summary(self):
        if not self.full_reports_buffer: return
        self.send_msg("⏳ <b>El Analista de Datos está procesando la información...</b>")
        
        # Unimos todos los reportes completos para darle contexto real a Gemini
        reports_text = "\n\n".join(self.full_reports_buffer)
        
        prompt = f"""
        Actúa como un Analista de Datos de Apuestas Deportivas (Sports Betting Data Scientist).
        
        A continuación, te pego varios reportes de partidos con estadísticas detalladas (X-RAY y PODER).
        Tu trabajo es ignorar la narrativa y centrarte exclusivamente en los datos duros.

        {reports_text}

        Tu Misión:
        1. Detectar el Valor Real: Cruza los datos de xG y Probabilidades del modelo (X-RAY). 
           Si el PICK sugerido por el bot es matemáticamente sólido, confírmalo. Si ves algo raro (ej: Pick Over pero xG muy bajo), aconséjalo.
        
        2. Formato Resumido: Dame una tabla o lista rápida:
           Partido ➡️ Mejor Apuesta (Data-Driven) ➡️ Razón de 5 palabras (ej: "xG alto", "Probabilidad 70%").
        
        3. El "Fun Parlay": Al final, genera una apuesta combinada (Parlay) lógica con TODOS los partidos listados. 
           Busca la opción más segura (aunque pague poco) para crear un ticket de alta probabilidad.

        USA SOLO negritas <b> y saltos de linea. NO uses Markdown (**).
        """
        try:
            ai_resp = self.call_gemini(prompt)
            self.send_msg(ai_resp)
        except Exception as e: self.send_msg(f"⚠️ Error Gemini: {e}")

    # --- BÚSQUEDA ---
    def find_team_in_global(self, team_name):
        if team_name in self.global_db: return self.global_db[team_name], team_name
        matches = difflib.get_close_matches(team_name, self.global_db.keys(), n=1, cutoff=0.6)
        if matches: return self.global_db[matches[0]], matches[0]
        return None, None

    # --- OUTPUT PROCESSOR ---
    def process_match_output(self, div, rh, ra, data, sim, best_bet, best_handi, today):
        is_cup = (div == 'EU_CUP')
        if not is_cup and div not in TOP_5_LEAGUES:
            return 
            
        if not best_bet: return
        
        is_valid = best_bet['status'] == "VALID"
        if is_valid:
            status_line = "✅ <b>PICK ACTIVO</b>"
            gcs_val = best_bet.get('gcs', 0)
            stake = self.get_stake(best_bet['prob'], best_bet['odd'], best_bet['market'], gcs_val)
            stake_txt = f"{stake*100:.2f}%"
            tag = "[VALID]"
            self.daily_picks_buffer.append(f"{tag} {rh} vs {ra}: {best_bet['pick']} @ {best_bet['odd']:.2f}")
        else:
            status_line = f"🚫 <b>NO BET</b> ({best_bet['reason']})"
            stake = 0.0; stake_txt = "Skipped"

        if best_handi:
            self.handicap_buffer.append(f"{rh} vs {ra}: {best_handi['pick']} @ {best_handi['odd']:.2f}")

        form_h = self.get_team_form_icon(data['raw_df'], rh)
        form_a = self.get_team_form_icon(data['raw_df'], ra) if 'raw_df' in data else "🛡️"
        
        ph, pd_raw, pa = sim['1x2']; dc1x, dcx2 = sim['dc']; dnb_h, dnb_a = sim['dnb']
        btts = sim['goals'][1]; ov25 = sim['goals'][0]; ah_h_m15, ah_a_m15, ah_h_p15, ah_a_p15 = sim['ah']
        h_stats, a_stats = sim['stats']; lambdas = sim['lambdas']; cs_str, cs_prob = sim['cs']
        fair_odd_us = self.dec_to_am(1/best_bet['prob']) if best_bet['prob'] > 0 else "-"
        pick_icon_display = "🎯" if is_valid else "⚠️"
        league_name = LEAGUE_CONFIG.get(div, {'name': '🏆 COPA EUROPA'})['name']

        msg = (
            f"🛡️ <b>ANÁLISIS v87</b> | {league_name}\n"
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
        
        # GUARDAR EN BUFFER PARA GEMINI (Esto es nuevo)
        if is_valid: self.full_reports_buffer.append(msg)
        
        if div in LEAGUE_CONFIG and div != 'EU_CUP':
            with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([today, div, rh, ra, best_bet['pick'], best_bet['market'], best_bet['prob'], best_bet['odd'], best_bet['ev'], best_bet['status'], stake, 0, "", ""])

    def run_analysis(self):
        self.run_audit()
        self.calculate_pnl()
        self.daily_picks_buffer = [] 
        self.full_reports_buffer = [] # Reset buffer
        self.handicap_buffer = []
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando v87.1 ANALYST: {today}", flush=True)
        
        print("🌍 Cargando DB Global...", flush=True)
        for div in LEAGUE_CONFIG:
            if div != 'EU_CUP': self.get_league_data(div)
        
        # 1. DOMESTICO
        ts = int(time.time())
        try:
            r = requests.get(f"https://www.football-data.co.uk/fixtures.csv?t={ts}", headers={'User-Agent': USER_AGENTS[0]}, timeout=20)
            if r.status_code==200:
                try: content = r.content.decode('utf-8-sig')
                except: content = r.content.decode('latin-1')
                df = pd.read_csv(io.StringIO(content), on_bad_lines='skip')
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

        # 2. COPAS
        if MANUAL_MATCHES:
            self.send_msg(f"🏆 <b>ANALIZANDO {len(MANUAL_MATCHES)} PARTIDOS DE COPA</b>")
            for home_input, away_input in MANUAL_MATCHES:
                h_data, real_h = self.find_team_in_global(home_input)
                a_data, real_a = self.find_team_in_global(away_input)
                if h_data and a_data:
                    hybrid_data = {
                        'stats': {real_h: {'att': h_data['att'], 'def': h_data['def']}, 
                                  real_a: {'att': a_data['att'], 'def': a_data['def']}},
                        'avg_g': (h_data['avg_g'] + a_data['avg_g']) / 2,
                        'inter_league': True,
                        'h_tier': h_data['tier'], 'a_tier': a_data['tier'],
                        'h_avg_g': h_data['avg_g'], 'a_avg_g': a_data['avg_g'],
                        'raw_df': h_data['raw_df']
                    }
                    sim = self.simulate_match(real_h, real_a, hybrid_data, {'H':0,'D':0,'A':0}, 0.5)
                    ph, pd, pa = sim['1x2']; p_o25 = sim['goals'][0]; p_btts = sim['goals'][1]
                    fair_odds = {
                        'H': 1/ph if ph>0 else 0, 'D': 1/pd if pd>0 else 0, 'A': 1/pa if pa>0 else 0,
                        'O25': 1/p_o25 if p_o25>0 else 0, 'BTTS_Y': 1/p_btts if p_btts>0 else 0
                    }
                    best_bet, best_handi = self.find_best_value(sim, fair_odds, -100)
                    self.process_match_output('EU_CUP', real_h, real_a, hybrid_data, sim, best_bet, best_handi, today)

        if self.daily_picks_buffer:
            self.generate_final_summary()
        else:
            self.send_msg("🧹 Barrido completado.")

if __name__ == "__main__":
    bot = OmniHybridBot()
    if os.getenv("SELF_TEST", "False") == "True": bot.run_analysis()
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: schedule.run_pending(); time.sleep(60)
