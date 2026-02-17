import pandas as pd
import numpy as np
import requests
import io
import difflib
import time
import schedule
import os
import csv
import re
import math
from datetime import datetime, timedelta
from collections import Counter

# --- CONFIGURACIÓN v88.0 (FULL DASHBOARD + MANUAL INJECTION) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "02:10" 
SEASON = '2526' 
SIMULATION_RUNS = 20000 # Volvemos a la simulación pesada para tener datos exactos

# --- ⚡ PARTIDOS MANUALES (CHAMPIONS/EUROPA) ⚡ ---
# Escribe aquí los partidos de hoy. El bot usará su DB global para simularlos.
MANUAL_MATCHES = [
    ('Galatasaray', 'Juventus'),
    ('Dortmund', 'Atalanta'),
    ('Monaco', 'PSG'),
    ('Benfica', 'Real Madrid'),
    ('Qarabag FK', 'Newcastle'),
    ('Olympiacos', 'Leverkusen'),
    ('Bodo/Glimt', 'Inter')
]

# --- 💾 PERSISTENCIA ---
VOLUME_PATH = "/app/data" 
HISTORY_FILE = os.path.join(VOLUME_PATH, "historial_omni_v88.csv") if os.path.exists(VOLUME_PATH) else "historial_omni_v88.csv"

# GESTIÓN DE RIESGO
KELLY_FRACTION = 0.20       
MAX_STAKE_PCT = 0.04        
USER_AGENTS = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36']

# --- CONFIGURACIÓN DE TIERS (Para Inferencia Inter-Ligas) ---
LEAGUE_CONFIG = {
    'E0':  {'name': '🇬🇧 PREMIER', 'tier': 1.00},
    'SP1': {'name': '🇪🇸 LA LIGA', 'tier': 1.00},
    'I1':  {'name': '🇮🇹 SERIE A', 'tier': 1.00},
    'D1':  {'name': '🇩🇪 BUNDES',  'tier': 1.00},
    'F1':  {'name': '🇫🇷 LIGUE 1', 'tier': 0.95},
    'P1':  {'name': '🇵🇹 PORTUGAL','tier': 0.85},
    'N1':  {'name': '🇳🇱 HOLANDA', 'tier': 0.85},
    'B1':  {'name': '🇧🇪 BELGICA', 'tier': 0.80},
    'T1':  {'name': '🇹🇷 TURQUIA', 'tier': 0.75},
    'SC0': {'name': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 ESCOCIA', 'tier': 0.70},
    'G1':  {'name': '🇬🇷 GRECIA',  'tier': 0.70}
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
        self.global_team_db = {} 
        self.daily_picks_buffer = [] 
        self.handicap_buffer = [] # Recuperado para el resumen
        
        print("--- ENGINE v88.0 FULL RESTORE STARTED ---", flush=True)
        self.send_msg("🔧 <b>INICIANDO v88.0</b>\n(Manual Injection + Full Dashboard Restored)")
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

    def send_msg(self, text):
        if not TELEGRAM_TOKEN: return
        # Sanitizer simple para evitar errores HTML
        text = text.replace("**", "").replace("```", "").strip()
        requests.post(f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)

    def call_gemini(self, prompt):
        if not SDK_AVAILABLE or not self.ai_client: return "❌ SDK no disponible."
        try:
            config = types.GenerateContentConfig(temperature=0.7)
            r = self.ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt, config=config)
            return r.text if r.text else "⚠️ Respuesta vacía."
        except Exception as e: return f"⚠️ Error Gemini: {str(e)[:100]}"

    # --- DATOS & ESTADÍSTICAS ---
    def calculate_team_stats(self, df, team):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(10)
        if len(matches) < 2: return 1.0, 1.0 
        
        w_att = 0; w_def = 0; total_w = 0
        for i, (_, row) in enumerate(matches.iterrows()):
            weight = 0.88 ** (len(matches) - 1 - i)
            total_w += weight
            if row['HomeTeam'] == team:
                w_att += row['FTHG'] * weight; w_def += row['FTAG'] * weight
            else:
                w_att += row['FTAG'] * weight; w_def += row['FTHG'] * weight
        
        if total_w == 0: return 1.0, 1.0
        return w_att / total_w, w_def / total_w

    def load_all_leagues(self):
        self.global_team_db = {}
        print("🌍 Cargando ecosistema de ligas...", flush=True)
        for div, config in LEAGUE_CONFIG.items():
            url = f"[https://www.football-data.co.uk/mmz4281/](https://www.football-data.co.uk/mmz4281/){SEASON}/{div}.csv"
            try:
                r = requests.get(url, headers={'User-Agent': USER_AGENTS[0]}, timeout=10)
                if r.status_code != 200: continue
                try: df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
                except: df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
                
                df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
                avg_g_league = (df.FTHG.mean() + df.FTAG.mean()) if not df.empty else 2.5
                teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
                
                league_avgs = {'a': 0, 'd': 0, 'c': 0}
                temp_stats = {}
                for t in teams:
                    a, d = self.calculate_team_stats(df, t)
                    temp_stats[t] = (a, d)
                    league_avgs['a'] += a; league_avgs['d'] += d; league_avgs['c'] += 1
                
                avg_a = league_avgs['a'] / league_avgs['c'] if league_avgs['c'] > 0 else 1
                avg_d = league_avgs['d'] / league_avgs['c'] if league_avgs['c'] > 0 else 1
                
                for t, (raw_a, raw_d) in temp_stats.items():
                    self.global_team_db[t] = {
                        'att': raw_a / avg_a, 'def': raw_d / avg_d, 'tier': config['tier'], 
                        'league_avg': avg_g_league, 'league_name': config['name'], 'raw_df': df
                    }
            except Exception as e: print(f"⚠️ Error {div}: {e}")
        print(f"✅ DB Global: {len(self.global_team_db)} equipos.", flush=True)

    def find_team_data(self, name):
        if name in self.global_team_db: return self.global_team_db[name], name
        match = difflib.get_close_matches(name, self.global_team_db.keys(), n=1, cutoff=0.6)
        return (self.global_team_db[match[0]], match[0]) if match else (None, None)

    def get_team_form_icon(self, df, team):
        # Iconos de forma recuperados
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
        pct = points / possible if possible > 0 else 0
        if pct >= 0.7: return "🔥"; 
        if pct <= 0.3: return "🧊"; 
        return "➡️"

    # --- SIMULADOR COMPLETO (Recuperado de v84) ---
    def calculate_dixon_coles_1x2(self, lambda_h, lambda_a):
        rho = -0.13; prob_h, prob_d, prob_a = 0.0, 0.0, 0.0
        for x in range(7):
            for y in range(7):
                p = (math.pow(lambda_h, x)*math.exp(-lambda_h)/math.factorial(x)) * (math.pow(lambda_a, y)*math.exp(-lambda_a)/math.factorial(y))
                if x==0 and y==0: p *= 1 - (lambda_h * lambda_a * rho)
                elif x==0 and y==1: p *= 1 + (lambda_h * rho)
                elif x==1 and y==0: p *= 1 + (lambda_a * rho)
                elif x==1 and y==1: p *= 1 - rho
                
                if x > y: prob_h += p
                elif x == y: prob_d += p
                else: prob_a += p
        total = prob_h + prob_d + prob_a
        return (prob_h/total, prob_d/total, prob_a/total) if total > 0 else (0.33, 0.33, 0.33)

    def simulate_match_full(self, home_input, away_input):
        h_data, h_name = self.find_team_data(home_input)
        a_data, a_name = self.find_team_data(away_input)
        if not h_data or not a_data: return None

        # Ajuste de Tier
        tier_diff = h_data['tier'] - a_data['tier']
        h_att = h_data['att'] * (1 + tier_diff * 0.45)
        h_def = h_data['def'] * (1 - tier_diff * 0.25)
        a_att = a_data['att'] * (1 - tier_diff * 0.45)
        a_def = a_data['def'] * (1 + tier_diff * 0.25)
        
        avg_g = (h_data['league_avg'] + a_data['league_avg']) / 2
        lambda_h = h_att * a_def * avg_g * 1.15
        lambda_a = a_att * h_def * avg_g
        
        # Simulación Monte Carlo (Completa)
        h_sim = np.random.poisson(lambda_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(lambda_a, SIMULATION_RUNS)
        
        prob_h, prob_d, prob_a = self.calculate_dixon_coles_1x2(lambda_h, lambda_a)
        
        over25 = np.mean((h_sim + a_sim) > 2.5)
        btts = np.mean((h_sim > 0) & (a_sim > 0))
        
        ah_h_minus = np.mean((h_sim - 1.5) > a_sim); ah_a_minus = np.mean((a_sim - 1.5) > h_sim)
        ah_h_plus = np.mean((h_sim + 1.5) > a_sim); ah_a_plus = np.mean((a_sim + 1.5) > h_sim)
        
        sim_scores = list(zip(h_sim, a_sim))
        most_common, count = Counter(sim_scores).most_common(1)[0]
        cs_str = f"{most_common[0]}-{most_common[1]}"
        cs_prob = (count / SIMULATION_RUNS) * 100
        
        return {
            'names': (h_name, a_name),
            '1x2': (prob_h, prob_d, prob_a),
            'goals': (over25, btts),
            'ah': (ah_h_minus, ah_a_minus, ah_h_plus, ah_a_plus),
            'lambdas': (lambda_h, lambda_a),
            'cs': (cs_str, cs_prob),
            'stats_raw': (h_data, a_data),
            'ev': 0.05 # Dummy EV para inyección manual (asumimos valor si lo pones manual)
        }

    # --- RESUMEN DE GEMINI (Recuperado de v84) ---
    def generate_final_summary(self):
        if not self.daily_picks_buffer and not self.handicap_buffer: return
        self.send_msg("⏳ <b>El Jefe de Estrategia está diseñando las jugadas maestras...</b>")
        
        picks_text = "\n".join(self.daily_picks_buffer)
        handi_text = "\n".join(self.handicap_buffer)
        
        prompt = f"""
        Actúa como Jefe de Estrategia de Apuestas de Fútbol.
        
        TUS PICKS HOY:
        {picks_text}
        
        OPCIONES SEGURAS (Handicaps):
        {handi_text}

        Genera un reporte breve y MOTIVADOR con:
        1. 💎 LA JOYA: (El mejor pick).
        2. 🛡️ EL BANKER: (El pick más seguro).
        3. 🎲 PARLAY RECOMENDADO: (Combina 2 opciones lógicas).
        
        Usa emojis. Sé directo. Usa negritas HTML <b>text</b>.
        """
        try:
            ai_resp = self.call_gemini(prompt)
            self.send_msg(ai_resp)
        except Exception as e: self.send_msg(f"⚠️ Gemini Error: {e}")

    # --- MAIN FLOW ---
    def run_analysis(self):
        self.load_all_leagues()
        self.daily_picks_buffer = []
        self.handicap_buffer = []
        
        self.send_msg(f"🏆 <b>ANALIZANDO {len(MANUAL_MATCHES)} PARTIDOS (MODO FULL)</b>")
        
        for h_in, a_in in MANUAL_MATCHES:
            sim = self.simulate_match_full(h_in, a_in)
            if not sim: 
                print(f"Skipping {h_in}-{a_in}"); continue
            
            real_h, real_a = sim['names']
            ph, pd, pa = sim['1x2']
            lh, la = sim['lambdas']
            cs_str, cs_prob = sim['cs']
            btts = sim['goals'][1]; ov25 = sim['goals'][0]
            ah_h_m15, ah_a_m15, ah_h_p15, ah_a_p15 = sim['ah']
            
            # Iconos de forma
            form_h = self.get_team_form_icon(sim['stats_raw'][0]['raw_df'], real_h)
            form_a = self.get_team_form_icon(sim['stats_raw'][1]['raw_df'], real_a)

            # Generar Pick Lógico (Sin cuotas reales, usamos probabilidad)
            pick = "SKIP"; odd_fair = 0; market = "-"
            
            # Lógica básica de pick para partidos manuales
            if ph > 0.45: pick = f"GANA {real_h}"; market="1X2"; odd_fair=1/ph
            elif pa > 0.45: pick = f"GANA {real_a}"; market="1X2"; odd_fair=1/pa
            elif ov25 > 0.55: pick = "OVER 2.5"; market="GOALS"; odd_fair=1/ov25
            elif btts > 0.58: pick = "BTTS SI"; market="BTTS"; odd_fair=1/btts
            else: pick = f"1X {real_h}" if ph > pa else f"X2 {real_a}"; market="DC"
            
            # Guardar para resumen
            self.daily_picks_buffer.append(f"{real_h} vs {real_a}: {pick} (Prob: {1/odd_fair*100:.0f}%)")
            if ah_h_p15 > 0.85: self.handicap_buffer.append(f"H +1.5 {real_h}")
            if ah_a_p15 > 0.85: self.handicap_buffer.append(f"H +1.5 {real_a}")

            # --- EL FORMATO DE SALIDA QUE QUERÍAS (RECUPERADO) ---
            msg = (
                f"🛡️ <b>ANÁLISIS v88.0</b> | 🇪🇺 CHAMPIONS/EUROPA\n"
                f"⚽ <b>{real_h}</b> {form_h} vs {form_a} <b>{real_a}</b>\n"
                f"───────────────\n"
                f"🎯 PICK: <b>{pick}</b> ({market})\n"
                f"🧠 Prob Modelo: <b>{1/odd_fair*100:.1f}%</b> (Fair: {odd_fair:.2f})\n"
                f"───────────────\n"
                f"📊 <b>X-RAY (Probabilidades):</b>\n"
                f"• 1X2: {ph*100:.0f}% | {pd*100:.0f}% | {pa*100:.0f}%\n"
                f"• BTTS: Sí {btts*100:.0f}% | No {(1-btts)*100:.0f}%\n"
                f"• Goals: Over {ov25*100:.0f}% | Under {(1-ov25)*100:.0f}%\n"
                f"• Handi -1.5: H {ah_h_m15*100:.0f}% | A {ah_a_m15*100:.0f}%\n"
                f"• Handi +1.5: H {ah_h_p15*100:.0f}% | A {ah_a_p15*100:.0f}%\n"
                f"───────────────\n"
                f"🎯 Marcador Probable: <b>{cs_str}</b> ({cs_prob:.1f}%)\n"
                f"⚔️ PODER (Exp.Goals):\n"
                f"🏠 {real_h}: <b>{lh:.2f}</b> gls\n"
                f"✈️ {real_a}: <b>{la:.2f}</b> gls"
            )
            self.send_msg(msg)
            time.sleep(1)
            
        # Generar Resumen Gemini
        self.generate_final_summary()

if __name__ == "__main__":
    bot = OmniHybridBot()
    # Ejecutar una vez al arrancar
    bot.run_analysis()
    
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: 
        schedule.run_pending()
        time.sleep(60)
