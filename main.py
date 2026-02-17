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

# --- CONFIGURACIÓN v87.2 (MANUAL INJECTION) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "01:52" 
SEASON = '2526' # Temporada

# --- ⚡ INYECCIÓN MANUAL DE PARTIDOS ⚡ ---
# Si el fixtures.csv no trae la Champions, escríbelos aquí.
# Formato: ('Local', 'Visitante')
# Usa los nombres en inglés o lo más parecido posible.
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
HISTORY_FILE = os.path.join(VOLUME_PATH, "historial_omni_v87.csv") if os.path.exists(VOLUME_PATH) else "historial_omni_v87.csv"

# GESTIÓN DE RIESGO
KELLY_FRACTION = 0.20       
MAX_STAKE_PCT = 0.04        
USER_AGENTS = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36']

# --- CONFIGURACIÓN DE TIERS (NIVEL DE LIGA) ---
# Esto permite al bot saber que un equipo de Premier es más fuerte que uno de Grecia
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
    SDK_AVAILABLE = True
except ImportError: pass

class OmniHybridBot:
    def __init__(self):
        self.global_team_db = {} 
        self.daily_picks_buffer = [] 
        print("--- ENGINE v87.2 HYBRID INJECTION STARTED ---", flush=True)
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
        text = text.replace("**", "").strip()
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)

    # --- DATOS & ESTADÍSTICAS ---
    def calculate_team_stats(self, df, team):
        # Analiza los últimos 10 partidos para sacar la fuerza de Ataque/Defensa
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(10)
        if len(matches) < 2: return 1.0, 1.0 # Pocos datos = promedio
        
        w_att = 0; w_def = 0; total_w = 0
        for i, (_, row) in enumerate(matches.iterrows()):
            weight = 0.88 ** (len(matches) - 1 - i) # Peso exponencial
            total_w += weight
            if row['HomeTeam'] == team:
                w_att += row['FTHG'] * weight
                w_def += row['FTAG'] * weight
            else:
                w_att += row['FTAG'] * weight
                w_def += row['FTHG'] * weight
        
        if total_w == 0: return 1.0, 1.0
        return w_att / total_w, w_def / total_w

    def load_all_leagues(self):
        self.global_team_db = {}
        print("🌍 Cargando ecosistema de ligas...", flush=True)
        
        for div, config in LEAGUE_CONFIG.items():
            url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{div}.csv"
            try:
                r = requests.get(url, headers={'User-Agent': USER_AGENTS[0]}, timeout=10)
                if r.status_code != 200: continue
                
                try: df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
                except: df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
                
                df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
                avg_g_league = (df.FTHG.mean() + df.FTAG.mean()) if not df.empty else 2.5
                
                teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
                
                # Calcular promedios de la liga para normalizar
                league_avgs = {'a': 0, 'd': 0, 'c': 0}
                temp_stats = {}
                
                for t in teams:
                    a, d = self.calculate_team_stats(df, t)
                    temp_stats[t] = (a, d)
                    league_avgs['a'] += a; league_avgs['d'] += d; league_avgs['c'] += 1
                
                avg_a = league_avgs['a'] / league_avgs['c'] if league_avgs['c'] > 0 else 1
                avg_d = league_avgs['d'] / league_avgs['c'] if league_avgs['c'] > 0 else 1
                
                # Guardar en DB Global
                for t, (raw_a, raw_d) in temp_stats.items():
                    self.global_team_db[t] = {
                        'att': raw_a / avg_a, 
                        'def': raw_d / avg_d, 
                        'tier': config['tier'], 
                        'league_avg': avg_g_league,
                        'league_name': config['name']
                    }
            except Exception as e: print(f"⚠️ Error {div}: {e}")
        
        print(f"✅ DB Global: {len(self.global_team_db)} equipos.", flush=True)

    def find_team_data(self, name):
        # Busca el equipo en la DB (Exacto o Aproximado)
        if name in self.global_team_db: return self.global_team_db[name], name
        match = difflib.get_close_matches(name, self.global_team_db.keys(), n=1, cutoff=0.6)
        return (self.global_team_db[match[0]], match[0]) if match else (None, None)

    # --- SIMULADOR INTER-LIGAS ---
    def simulate_match(self, home_input, away_input):
        h_data, h_name = self.find_team_data(home_input)
        a_data, a_name = self.find_team_data(away_input)
        
        if not h_data or not a_data:
            print(f"❌ No encontré datos para {home_input} o {away_input}")
            return None

        # Ajuste por Tier (Nivel de Liga)
        # Si Tier Local > Tier Visitante, aumentamos ataque Local y reducimos defensa Visitante
        tier_diff = h_data['tier'] - a_data['tier']
        
        # Factores de corrección
        h_att = h_data['att'] * (1 + tier_diff * 0.45)
        h_def = h_data['def'] * (1 - tier_diff * 0.25)
        
        a_att = a_data['att'] * (1 - tier_diff * 0.45) # Inverso
        a_def = a_data['def'] * (1 + tier_diff * 0.25)
        
        avg_goals = (h_data['league_avg'] + a_data['league_avg']) / 2
        
        # Lambdas (Goles esperados)
        lambda_h = h_att * a_def * avg_goals * 1.15 # 15% Ventaja localía europea
        lambda_a = a_att * h_def * avg_goals
        
        # Probabilidades Poisson
        prob_h, prob_d, prob_a = self.calculate_probs(lambda_h, lambda_a)
        
        return {
            'names': (h_name, a_name),
            '1x2': (prob_h, prob_d, prob_a),
            'lambdas': (lambda_h, lambda_a),
            'tiers': (h_data['tier'], a_data['tier'])
        }

    def calculate_probs(self, lh, la):
        # Dixon-Coles simple
        ph, pd, pa = 0, 0, 0
        for x in range(7):
            for y in range(7):
                p = (math.pow(lh, x)*math.exp(-lh)/math.factorial(x)) * (math.pow(la, y)*math.exp(-la)/math.factorial(y))
                if x > y: ph += p
                elif x == y: pd += p
                else: pa += p
        return ph, pd, pa

    def run_analysis(self):
        self.load_all_leagues()
        
        if not MANUAL_MATCHES:
            self.send_msg("⚠️ No hay partidos manuales configurados en MANUAL_MATCHES.")
            return

        self.send_msg(f"🏆 <b>ANALIZANDO {len(MANUAL_MATCHES)} PARTIDOS DE COPA</b>")
        
        for h_in, a_in in MANUAL_MATCHES:
            sim = self.simulate_match(h_in, a_in)
            if not sim: continue
            
            real_h, real_a = sim['names']
            ph, pd, pa = sim['1x2']
            lh, la = sim['lambdas']
            
            # Generar Pick basado en probabilidad
            pick = ""; odd_fair = 0
            if ph > 0.45: pick = f"GANA {real_h}"; odd_fair = 1/ph
            elif pa > 0.40: pick = f"GANA {real_a}"; odd_fair = 1/pa
            elif (ph+pd) > 0.75: pick = f"1X {real_h}"; odd_fair = 1/(ph+pd)
            else: pick = "NO BET (Muy parejo)"
            
            # Detectar Over/Under
            total_xg = lh + la
            goals_pick = "OVER 2.5" if total_xg > 2.75 else ("UNDER 2.5" if total_xg < 2.2 else "PASS")
            
            msg = (
                f"⚽ <b>{real_h} vs {real_a}</b>\n"
                f"📊 <b>Stats:</b>\n"
                f"• Goles Esp: {lh:.2f} - {la:.2f}\n"
                f"• Prob Gana: {ph*100:.0f}% - {pa*100:.0f}%\n"
                f"• Tier: {sim['tiers'][0]} vs {sim['tiers'][1]}\n"
                f"───────────────\n"
                f"🧠 <b>PROYECCIÓN:</b>\n"
                f"🎯 1X2: <b>{pick}</b> (Fair: {odd_fair:.2f})\n"
                f"🥅 Goles: <b>{goals_pick}</b>"
            )
            self.send_msg(msg)
            time.sleep(1)

if __name__ == "__main__":
    bot = OmniHybridBot()
    # Ejecutar directamente al iniciar
    bot.run_analysis()
    
    # Mantener vivo si es necesario
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: 
        schedule.run_pending()
        time.sleep(60)
