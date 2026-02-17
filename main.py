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

# --- CONFIGURACIÓN v87.0 (ZERO-TOUCH / INTER-LEAGUE) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "01:46" 

# AJUSTES DE MODELO
SIMULATION_RUNS = 20000 
DECAY_ALPHA = 0.88          
SEASON = '2526' # Ajustar año (ej: 2425 para 2024/2025)

# --- 💾 PERSISTENCIA ---
VOLUME_PATH = "/app/data" 
if os.path.exists(VOLUME_PATH):
    HISTORY_FILE = os.path.join(VOLUME_PATH, "historial_omni_v87.csv")
else:
    HISTORY_FILE = "historial_omni_v87.csv"

# GESTIÓN DE RIESGO
KELLY_FRACTION = 0.20       
MAX_STAKE_PCT = 0.04        

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

# --- CONFIGURACIÓN DE LIGAS (SOLO LAS QUE TIENEN CSV AUTOMÁTICO) ---
# El bot usará estas ligas para inferir la fuerza en Copas Europeas
LEAGUE_CONFIG = {
    'E0':  {'name': '🇬🇧 PREMIER', 'tier': 1.00, 'm_weight': 0.85, 'min_ev': 0.02},
    'SP1': {'name': '🇪🇸 LA LIGA', 'tier': 1.00, 'm_weight': 0.85, 'min_ev': 0.02},
    'I1':  {'name': '🇮🇹 SERIE A', 'tier': 1.00, 'm_weight': 0.82, 'min_ev': 0.02},
    'D1':  {'name': '🇩🇪 BUNDES',  'tier': 1.00, 'm_weight': 0.82, 'min_ev': 0.02},
    'F1':  {'name': '🇫🇷 LIGUE 1', 'tier': 0.95, 'm_weight': 0.80, 'min_ev': 0.025},
    'P1':  {'name': '🇵🇹 PORTUGAL','tier': 0.85, 'm_weight': 0.70, 'min_ev': 0.03},
    'N1':  {'name': '🇳🇱 HOLANDA', 'tier': 0.85, 'm_weight': 0.70, 'min_ev': 0.03},
    'B1':  {'name': '🇧🇪 BELGICA', 'tier': 0.80, 'm_weight': 0.65, 'min_ev': 0.04},
    'T1':  {'name': '🇹🇷 TURQUIA', 'tier': 0.75, 'm_weight': 0.60, 'min_ev': 0.04},
    'SC0': {'name': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 ESCOCIA', 'tier': 0.70, 'm_weight': 0.60, 'min_ev': 0.04},
    'G1':  {'name': '🇬🇷 GRECIA',  'tier': 0.70, 'm_weight': 0.60, 'min_ev': 0.04}
}

# Códigos de Copas en el archivo de Fixtures (Esto permite identificar el torneo)
CUP_CODES = ['E1', 'E2', 'EC', 'UCL', 'UEL', 'UECL'] 

# --- DIAGNÓSTICO ---
SDK_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
except ImportError: pass

class OmniHybridBot:
    def __init__(self):
        self.global_team_db = {} # Base de datos de equipos en memoria
        self.daily_picks_buffer = [] 
        self.handicap_buffer = [] 
        
        print("--- ENGINE v87.0 INTER-LEAGUE STARTED ---", flush=True)
        self.send_msg(f"🔧 <b>INICIANDO v87.0 (Zero-Touch)</b>\nModo: Inter-League Inference")
        self._init_history_file()
        
        self.ai_client = None
        if SDK_AVAILABLE and GEMINI_API_KEY:
            try:
                self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
            except: pass

    def _init_history_file(self):
        if not os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Date', 'League', 'Home', 'Away', 'Pick', 'Market', 'Prob', 'Odd', 'EV', 'Status', 'Stake', 'Profit', 'FTHG', 'FTAG'])
            except: pass

    def send_msg(self, text, retry_count=0, use_html=True):
        if not TELEGRAM_TOKEN: return
        # Limpieza simple
        if use_html: text = text.replace("**", "").strip()
        
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML" if use_html else None}
        try: requests.post(url, json=payload, timeout=10)
        except: pass

    def calculate_team_stats(self, df, team):
        # Cálculo clásico exponencial
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(10) # Usamos más historial para estabilidad
        if len(matches) < 3: return 1.0, 1.0
        
        w_att = 0; w_def = 0; total_w = 0
        for i, (_, row) in enumerate(matches.iterrows()):
            weight = pow(DECAY_ALPHA, len(matches) - 1 - i)
            total_w += weight
            
            if row['HomeTeam'] == team:
                att = row['FTHG']; def_weak = row['FTAG']
            else:
                att = row['FTAG']; def_weak = row['FTHG']
                
            w_att += att * weight; w_def += def_weak * weight
        
        if total_w == 0: return 1.0, 1.0
        return w_att / total_w, w_def / total_w

    def load_all_leagues(self):
        """Carga TODAS las ligas configuradas en una DB global para uso cruzado"""
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
                avg_g = (df.FTHG.mean() + df.FTAG.mean()) if not df.empty else 2.5
                
                teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
                
                # Normalización de la liga específica
                league_stats = {}
                avg_a_league = 0; avg_d_league = 0; count = 0
                
                for t in teams:
                    a, d = self.calculate_team_stats(df, t)
                    league_stats[t] = {'raw_att': a, 'raw_def': d}
                    avg_a_league += a; avg_d_league += d; count += 1
                
                if count > 0: avg_a_league /= count; avg_d_league /= count
                else: avg_a_league = 1; avg_d_league = 1
                
                # Guardar en DB Global con el Tier de la liga
                tier_factor = config['tier']
                for t, s in league_stats.items():
                    # Normalizamos stats relativas a su liga
                    norm_att = s['raw_att'] / avg_a_league
                    norm_def = s['raw_def'] / avg_d_league
                    
                    self.global_team_db[t] = {
                        'att': norm_att, 
                        'def': norm_def, 
                        'tier': tier_factor, 
                        'league': div,
                        'avg_g_league': avg_g,
                        'form_df': df # Guardamos referencia para iconos de forma
                    }
                    
            except Exception as e:
                print(f"Error cargando {div}: {e}")
        
        print(f"✅ DB Global lista: {len(self.global_team_db)} equipos indexados.", flush=True)

    def find_team_stats(self, team_name):
        # Búsqueda difusa en la base de datos global
        if not self.global_team_db: return None
        
        # 1. Match exacto
        if team_name in self.global_team_db:
            return self.global_team_db[team_name], team_name
            
        # 2. Match aproximado
        all_teams = list(self.global_team_db.keys())
        matches = difflib.get_close_matches(team_name, all_teams, n=1, cutoff=0.65)
        
        if matches:
            found_name = matches[0]
            return self.global_team_db[found_name], found_name
            
        return None, None

    def simulate_inter_league_match(self, home_name, away_name, market_odds):
        h_data, h_real_name = self.find_team_stats(home_name)
        a_data, a_real_name = self.find_team_stats(away_name)
        
        if not h_data or not a_data:
            return None # No encontramos datos suficientes
            
        # --- LÓGICA DE AJUSTE INTER-LIGAS ---
        # Si el Home es de Tier 1 (Premier) y Away de Tier 0.7 (Grecia), 
        # el ataque de Home debe valer más y su defensa ser más fuerte.
        
        h_tier = h_data['tier']
        a_tier = a_data['tier']
        
        # Diferencia de nivel
        tier_diff = h_tier - a_tier # Positivo si H es mejor liga
        
        # Factores de ajuste (Empíricos)
        # Un equipo promedio de Premier es aprox 1.3x más fuerte que uno de Grecia
        tier_adjust_att = 1 + (tier_diff * 0.4) 
        tier_adjust_def = 1 - (tier_diff * 0.2)
        
        h_att = h_data['att'] * tier_adjust_att
        h_def = h_data['def'] * tier_adjust_def
        
        # Para el visitante, la lógica inversa
        a_att = a_data['att'] * (1 / tier_adjust_att) # Si H es mejor, A ataca peor
        a_def = a_data['def'] * (1 / tier_adjust_def) # Si H es mejor, A defiende peor
        
        # Promedio global de gol (promedio de ambas ligas)
        avg_g_match = (h_data['avg_g_league'] + a_data['avg_g_league']) / 2
        
        # Lambdas Dixon-Coles
        lambda_h = h_att * a_def * (avg_g_match / 2) * 1.15 # Factor localía europeo
        lambda_a = a_att * h_def * (avg_g_match / 2)
        
        # Simulación
        prob_h, prob_d, prob_a = self.calculate_dixon_coles_1x2(lambda_h, lambda_a)
        
        # Ajuste con mercado
        if market_odds['H'] > 0:
            margin = 1.06
            implied_h = (1/market_odds['H']) / margin
            implied_a = (1/market_odds['A']) / margin
            implied_d = (1/market_odds['D']) / margin
            
            # En copas confiamos más en el mercado que en nuestro modelo inferido
            w_market = 0.65 
            
            prob_h = (prob_h * (1-w_market)) + (implied_h * w_market)
            prob_a = (prob_a * (1-w_market)) + (implied_a * w_market)
            prob_d = (prob_d * (1-w_market)) + (implied_d * w_market)
            
            # Renormalizar
            total = prob_h + prob_d + prob_a
            prob_h /= total; prob_a /= total; prob_d /= total

        # Goals calc
        h_sim = np.random.poisson(lambda_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(lambda_a, SIMULATION_RUNS)
        over25 = np.mean((h_sim + a_sim) > 2.5)
        btts = np.mean((h_sim > 0) & (a_sim > 0))
        
        # Calidad de predicción (EV Score)
        ev_quality = 70 # Base standard para cups
        
        return {
            'names': (h_real_name, a_real_name),
            '1x2': (prob_h, prob_d, prob_a),
            'goals': (over25, btts),
            'gcs': ev_quality,
            'stats': (h_data, a_data),
            'lambdas': (lambda_h, lambda_a)
        }

    # --- MATEMÁTICAS CORE ---
    def poisson_prob(self, k, lamb):
        return (math.pow(lamb, k) * math.exp(-lamb)) / math.factorial(k)

    def calculate_dixon_coles_1x2(self, lambda_h, lambda_a):
        rho = -0.13; prob_h, prob_d, prob_a = 0.0, 0.0, 0.0
        for x in range(7):
            for y in range(7):
                p = self.poisson_prob(x, lambda_h) * self.poisson_prob(y, lambda_a)
                if x==0 and y==0: p *= 1 - (lambda_h * lambda_a * rho)
                elif x==0 and y==1: p *= 1 + (lambda_h * rho)
                elif x==1 and y==0: p *= 1 + (lambda_a * rho)
                elif x==1 and y==1: p *= 1 - rho
                
                if x > y: prob_h += p
                elif x == y: prob_d += p
                else: prob_a += p
        total = prob_h + prob_d + prob_a
        return (prob_h/total, prob_d/total, prob_a/total) if total > 0 else (0.33, 0.33, 0.33)

    def get_avg_odds(self, row):
        def get_avg(cols):
            vals = [float(row[c]) for c in cols if row.get(c) and str(row[c]).replace('.','').isdigit()]
            return sum(vals)/len(vals) if vals else 0.0
        return {
            'H': get_avg(['B365H', 'PSH', 'WHH']), 'D': get_avg(['B365D', 'PSD', 'WHD']),
            'A': get_avg(['B365A', 'PSA', 'WHA']), 'O25': get_avg(['B365>2.5', 'P>2.5', 'WH>2.5']),
            'BTTS_Y': get_avg(['BbAvBBTS', 'B365BTTSY'])
        }

    def find_best_value(self, sim, odds):
        # Lógica simplificada para modo Híbrido
        probs = sim['1x2']
        candidates = []
        
        # 1X2
        if odds['H'] > 1.1:
            ev = (probs[0] * odds['H']) - 1
            if ev > 0.05: candidates.append({'pick': 'GANA LOCAL', 'm': '1X2', 'odd': odds['H'], 'ev': ev, 'p': probs[0]})
            
        if odds['A'] > 1.1:
            ev = (probs[2] * odds['A']) - 1
            if ev > 0.05: candidates.append({'pick': 'GANA VISITANTE', 'm': '1X2', 'odd': odds['A'], 'ev': ev, 'p': probs[2]})
            
        # Goals
        if odds['O25'] > 1.1:
            ev = (sim['goals'][0] * odds['O25']) - 1
            if ev > 0.04: candidates.append({'pick': 'OVER 2.5', 'm': 'GOALS', 'odd': odds['O25'], 'ev': ev, 'p': sim['goals'][0]})
            
        # Ordenar por EV
        candidates.sort(key=lambda x: x['ev'], reverse=True)
        return candidates[0] if candidates else None

    # --- MAIN FLOW ---
    def run_analysis(self):
        self.daily_picks_buffer = []
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando v87.0 (Auto-Discovery): {today}", flush=True)
        
        # 1. Cargar datos frescos de todas las ligas configuradas
        self.load_all_leagues()
        
        # 2. Descargar Fixtures (Partidos de hoy/mañana)
        try:
            ts = int(time.time())
            url_fixt = f"https://www.football-data.co.uk/fixtures.csv?t={ts}"
            r = requests.get(url_fixt, headers={'User-Agent': USER_AGENTS[0]}, timeout=15)
            df_fixt = pd.read_csv(io.StringIO(r.content.decode('latin-1')), on_bad_lines='skip')
            df_fixt['Date'] = pd.to_datetime(df_fixt['Date'], dayfirst=True, errors='coerce')
        except:
            print("❌ Error descargando fixtures")
            return

        target_date = pd.to_datetime(today, dayfirst=True)
        daily = df_fixt[(df_fixt['Date'] >= target_date) & (df_fixt['Date'] <= target_date + timedelta(days=1))]
        
        self.send_msg(f"🔎 <b>Analizando {len(daily)} partidos hoy...</b>")
        
        bets_found = 0
        
        for idx, row in daily.iterrows():
            div = row.get('Div')
            home = row['HomeTeam']
            away = row['AwayTeam']
            m_odds = self.get_avg_odds(row)
            
            if m_odds['H'] == 0: continue # Sin cuotas
            
            # ESTRATEGIA:
            # Si la liga está en LEAGUE_CONFIG, usa el análisis normal (Tier 1 accuracy).
            # Si la liga NO está (ej: E1, EC, copas), usa el MODO HÍBRIDO buscando equipos en la DB Global.
            
            sim_result = None
            is_cup_match = False
            
            if div in LEAGUE_CONFIG:
                # Análisis normal (mismo código que v85)
                # (Simplificado aquí para usar la misma función de simulación híbrida que funciona bien)
                sim_result = self.simulate_inter_league_match(home, away, m_odds)
            elif div in CUP_CODES or any(c in str(div) for c in CUP_CODES):
                # ES UN PARTIDO DE COPA
                is_cup_match = True
                sim_result = self.simulate_inter_league_match(home, away, m_odds)
            
            if sim_result:
                best = self.find_best_value(sim_result, m_odds)
                
                if best and best['ev'] > 0.05: # Umbral EV
                    bets_found += 1
                    real_h, real_a = sim_result['names']
                    
                    # Formateo de nombres de copas
                    league_name = LEAGUE_CONFIG[div]['name'] if div in LEAGUE_CONFIG else "🇪🇺 COPA EUROPA"
                    
                    # Kelly simple
                    kelly = (best['ev'] / (best['odd'] - 1)) * 0.2
                    stake = min(max(0, kelly), 0.04)
                    
                    msg = (
                        f"🎯 <b>PICK DETECTADO ({'🏆 COPA' if is_cup_match else 'LIGA'})</b>\n"
                        f"⚽ {real_h} vs {real_a}\n"
                        f"🏆 {league_name}\n"
                        f"───────────────\n"
                        f"✅ <b>{best['pick']}</b>\n"
                        f"📈 Cuota: {best['odd']:.2f} | EV: {best['ev']*100:.1f}%\n"
                        f"💰 Stake Sugerido: {stake*100:.1f}%\n"
                        f"🧠 Prob Modelo: {best['p']*100:.1f}%"
                    )
                    self.send_msg(msg)
                    self.daily_picks_buffer.append(f"{real_h} v {real_a}: {best['pick']} @ {best['odd']}")

                    # Guardar historial
                    with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerow([today, div, real_h, real_a, best['pick'], best['m'], best['p'], best['odd'], best['ev'], "VALID", stake, 0, "", ""])

        if bets_found == 0:
            self.send_msg("🧹 Barrido completado: Sin valor claro hoy.")
        elif self.ai_client:
            # Resumen Gemini opcional
            try:
                txt = "\n".join(self.daily_picks_buffer)
                resp = self.ai_client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=f"Resume estos picks de fútbol en un tono motivador:\n{txt}"
                )
                self.send_msg(f"🎙️ <b>Comentario Técnico:</b>\n{resp.text}")
            except: pass

if __name__ == "__main__":
    bot = OmniHybridBot()
    if os.getenv("SELF_TEST", "False") == "True": 
        bot.run_analysis()
    
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: 
        schedule.run_pending()
        time.sleep(60)
