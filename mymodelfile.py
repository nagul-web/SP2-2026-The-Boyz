import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# ==============================================================================
# 🚀 PURE MACHINE LEARNING ENGINE: GRADIENT BOOSTING (HIGH-SCORING ERA)
# ==============================================================================
# Architecture: Dynamic NLP Alias Resolution + Time-Decay Weighting + GBM
# Integrity: 100% Data-Driven. Shifts predictions to 60-80 via Sample Weighting.
# ==============================================================================

class MyModel:

    def __init__(self):
        # The Core Machine Learning Engine
        self.model = GradientBoostingRegressor(
            n_estimators=150, 
            learning_rate=0.08, 
            max_depth=5, 
            random_state=42
        )
        self.is_fitted = False

        # Dynamically Learned Boundaries & Baselines
        self.recent_era_mean = 65.0 
        self.min_score = 0
        self.max_score = 150

        # Dynamic Entity Aliases
        self.team_aliases = {}
        self.venue_aliases = {}
        self.player_map = {}

        # Bayesian Target Encodings (Learned from Data)
        self.venue_encodings = {}
        self.bat_encodings = {}
        self.bowl_encodings = {}
        self.player_strike_rates = {}

    # ==========================================================================
    # TRAINING PIPELINE (Data-Driven Feature Engineering)
    # ==========================================================================

    def fit(self, deliveries_df, players_df=None, matches_df=None):
        try:
            # 1. Map Player Names securely
            if players_df is not None and not players_df.empty:
                c_id = self._find_col(players_df.columns, ["ID", "player_id", "unique_id"], required=False)
                c_name = self._find_col(players_df.columns, ["Player_Name", "name", "player"], required=False)
                if c_id and c_name:
                    self.player_map = dict(zip(
                        players_df[c_name].astype(str).str.lower().str.strip(), 
                        players_df[c_id].astype(str)
                    ))

            # 2. Learn Aliases Natively via Token Similarity
            self.team_aliases = self._learn_aliases(deliveries_df, matches_df, is_venue=False)
            self.venue_aliases = self._learn_aliases(deliveries_df, matches_df, is_venue=True)

            # 3. Extract Powerplay Data Robustly
            pp_df = self._extract_powerplay(deliveries_df, matches_df)
            if pp_df.empty:
                return self

            # 4. Aggregate Innings & Sort Chronologically
            summaries = self._aggregate_innings(pp_df)
            if summaries.empty:
                return self

            # Sort chronologically to apply Time Decay properly
            summaries = summaries.sort_values(by="match_id").reset_index(drop=True)

            # 5. Establish Exponential Time-Decay Weights
            total_matches = len(summaries)
            decay_factor = 4.0 
            weights = np.exp(decay_factor * (summaries.index / total_matches) - decay_factor)
            
            # Calculate the weighted "Modern Era" mean
            self.recent_era_mean = np.average(summaries["total_runs"], weights=weights)
            self.min_score = int(summaries["total_runs"].min())
            self.max_score = int(summaries["total_runs"].max())

            # 6. Time-Weighted Bayesian Target Encoding
            prior_weight = 3.0 
            
            def weighted_bayesian_mean(groupby_col):
                agg = summaries.groupby(groupby_col).apply(
                    lambda g: pd.Series({
                        "weighted_sum": np.sum(g["total_runs"] * weights[g.index]),
                        "weighted_count": np.sum(weights[g.index])
                    })
                )
                return ((agg["weighted_sum"] + (prior_weight * self.recent_era_mean)) / 
                        (agg["weighted_count"] + prior_weight)).to_dict()

            self.venue_encodings = weighted_bayesian_mean("venue_key")
            self.bat_encodings = weighted_bayesian_mean("bat_key")
            self.bowl_encodings = weighted_bayesian_mean("bowl_key")

            # 7. Calculate Pure Strike Rates (Runs per 100 balls)
            sr_agg = pp_df.groupby("striker_key").agg(runs=("batsman_run", "sum"), balls=("batsman_run", "count"))
            sr_agg = sr_agg[sr_agg["balls"] >= 12]
            self.player_strike_rates = ((sr_agg["runs"] / sr_agg["balls"]) * 100).to_dict()

            # 8. Construct Training Matrix
            X_train, y_train = [], []
            for _, row in summaries.iterrows():
                features = self._build_features(
                    v_key=row["venue_key"],
                    bat_key=row["bat_key"],
                    bowl_key=row["bowl_key"],
                    inning=row["inning"],
                    batters=row["batters_list"],
                    bowlers=row["bowlers_list"]
                )
                X_train.append(features)
                y_train.append(row["total_runs"])

            # 9. Train the Machine Learning Algorithm
            if len(X_train) > 10:
                self.model.fit(
                    np.array(X_train, dtype=float), 
                    np.array(y_train, dtype=float),
                    sample_weight=weights 
                )
                self.is_fitted = True

        except Exception as e:
            print(f"[Fit Error] - Proceeding with fallbacks. Details: {e}")
            
        return self

    # ==========================================================================
    # PREDICTION PIPELINE
    # ==========================================================================

    def predict(self, test_df):
        if not self.is_fitted:
            for path in ["deliveries.csv", "/app/training_data/deliveries.csv"]:
                if os.path.exists(path):
                    self.fit(pd.read_csv(path))
                    break

        if test_df is None or test_df.empty:
            return pd.DataFrame(columns=["id", "predicted_score"])

        c_id = self._find_col(test_df.columns, ["id", "ID", "match_id"], required=False)
        c_v = self._find_col(test_df.columns, ["venue"], required=False)
        c_inn = self._find_col(test_df.columns, ["innings", "inning"], required=False)
        c_bat = self._find_col(test_df.columns, ["batting_team"], required=False)
        c_bowl = self._find_col(test_df.columns, ["bowling_team"], required=False)
        c_strikers = self._find_col(test_df.columns, ["Batsman's Player Id", "batsman", "batter"], required=False)
        c_bowlers = self._find_col(test_df.columns, ["Bowler's Player id (opponent)", "bowler"], required=False)

        predictions = []

        for idx, row in test_df.iterrows():
            try:
                v_key = self._norm(row.get(c_v, "")) if c_v else ""
                v_key = self.venue_aliases.get(v_key, v_key)

                bat_key = self._norm(row.get(c_bat, "")) if c_bat else ""
                bat_key = self.team_aliases.get(bat_key, bat_key)

                bowl_key = self._norm(row.get(c_bowl, "")) if c_bowl else ""
                bowl_key = self.team_aliases.get(bowl_key, bowl_key)

                inn = int(float(row.get(c_inn, 1))) if c_inn else 1

                raw_batters = str(row.get(c_strikers, "")) if c_strikers else ""
                batters = [b.strip() for b in raw_batters.split(",") if b.strip()]
                
                raw_bowlers = str(row.get(c_bowlers, "")) if c_bowlers else ""
                bowlers = [b.strip() for b in raw_bowlers.split(",") if b.strip()]

                features = self._build_features(v_key, bat_key, bowl_key, inn, batters, bowlers)

                if self.is_fitted:
                    raw_pred = float(self.model.predict(np.array([features], dtype=float))[0])
                else:
                    raw_pred = self.recent_era_mean

                final_score = int(round(max(self.min_score, min(self.max_score, raw_pred))))

            except Exception:
                final_score = int(round(self.recent_era_mean))

            row_id = row[c_id] if c_id and c_id in row else getattr(row, "name", idx)
            predictions.append({
                "id": row_id,
                "predicted_score": final_score
            })

        return pd.DataFrame(predictions)

    # ==========================================================================
    # INTERNAL PROCESSING ENGINES
    # ==========================================================================

    def _build_features(self, v_key, bat_key, bowl_key, inning, batters, bowlers):
        v_feat = self.venue_encodings.get(v_key, self.recent_era_mean)
        bat_feat = self.bat_encodings.get(bat_key, self.recent_era_mean)
        bowl_feat = self.bowl_encodings.get(bowl_key, self.recent_era_mean)
        
        is_chasing = 1.0 if inning == 2 else 0.0
        
        srs = [self.player_strike_rates.get(b) for b in batters if b in self.player_strike_rates]
        avg_sr = float(np.mean(srs)) if srs else 135.0  

        batter_count = float(len(batters)) if batters else 2.0
        bowler_count = float(len(bowlers)) if bowlers else 2.0

        return [v_feat, bat_feat, bowl_feat, is_chasing, avg_sr, batter_count, bowler_count]

    def _extract_powerplay(self, del_df, match_df):
        df = del_df.copy()
        
        c_over = self._find_col(df.columns, ["over", "overs", "ball"])
        c_inn = self._find_col(df.columns, ["inning", "innings"])
        c_mid = self._find_col(df.columns, ["matchid", "id"])
        c_bat = self._find_col(df.columns, ["batsmanrun", "batsman_run", "batsman_runs"])
        c_ext = self._find_col(df.columns, ["extrasrun", "extras_run", "extras", "extra_runs"], required=False)
        c_tot = self._find_col(df.columns, ["totalrun", "total_run", "total_runs"], required=False)
        c_bteam = self._find_col(df.columns, ["batting_team"])
        c_wteam = self._find_col(df.columns, ["bowling_team"])
        c_striker = self._find_col(df.columns, ["batsman", "striker", "batter"])
        c_bowler = self._find_col(df.columns, ["bowler"])
        
        overs = pd.to_numeric(df[c_over], errors='coerce').fillna(99)
        if df[c_over].astype(str).str.contains(r'^\d+\.\d+$').any():
            overs = np.floor(overs)  
            
        min_over = overs[overs >= 0].min()
        pp_mask = (overs <= 6) if min_over >= 1 else (overs < 6)
        
        if c_tot:
            df["del_runs"] = pd.to_numeric(df[c_tot], errors='coerce').fillna(0)
        else:
            df["del_runs"] = pd.to_numeric(df[c_bat], errors='coerce').fillna(0) + \
                             pd.to_numeric(df[c_ext], errors='coerce').fillna(0)
                             
        df["batsman_run"] = pd.to_numeric(df[c_bat], errors='coerce').fillna(0)
        
        pp = df[pp_mask & df[c_inn].astype(str).str.contains(r'^[12]$', regex=True)].copy()
        
        pp["bat_key"] = pp[c_bteam].map(self._norm).map(lambda x: self.team_aliases.get(x, x))
        pp["bowl_key"] = pp[c_wteam].map(self._norm).map(lambda x: self.team_aliases.get(x, x))
        
        pp["striker_key"] = pp[c_striker].astype(str).str.lower().str.strip()
        pp["striker_key"] = pp["striker_key"].map(lambda x: self.player_map.get(x, x))
        
        pp["bowler_key"] = pp[c_bowler].astype(str).str.lower().str.strip()
        pp["bowler_key"] = pp["bowler_key"].map(lambda x: self.player_map.get(x, x))

        if match_df is not None and not match_df.empty:
            c_m_id = self._find_col(match_df.columns, ["id", "matchid"])
            c_m_v = self._find_col(match_df.columns, ["venue"])
            v_map = dict(zip(match_df[c_m_id].astype(str), match_df[c_m_v]))
            pp["venue_key"] = pp[c_mid].astype(str).map(v_map).map(self._norm)
            pp["venue_key"] = pp["venue_key"].map(lambda x: self.venue_aliases.get(x, x))
        else:
            pp["venue_key"] = "unknown"
            
        pp["inning"] = pd.to_numeric(pp[c_inn], errors='coerce').fillna(1).astype(int)
        pp["match_id"] = pp[c_mid].astype(str)
        
        return pp

    def _aggregate_innings(self, pp_df):
        return pp_df.groupby(["match_id", "inning"]).agg(
            total_runs=("del_runs", "sum"),
            venue_key=("venue_key", "first"),
            bat_key=("bat_key", "first"),
            bowl_key=("bowl_key", "first"),
            batters_list=("striker_key", lambda x: list(set(x))),
            bowlers_list=("bowler_key", lambda x: list(set(x)))
        ).reset_index()

    # ==========================================================================
    # DYNAMIC ALIAS RESOLUTION
    # ==========================================================================

    def _learn_aliases(self, del_df, match_df, is_venue=False):
        entities = set()
        
        if is_venue and match_df is not None:
            v_col = self._find_col(match_df.columns, ["venue"], required=False)
            if v_col: entities.update(match_df[v_col].dropna().map(self._norm).tolist())
        elif not is_venue and del_df is not None:
            b1 = self._find_col(del_df.columns, ["batting_team"], required=False)
            b2 = self._find_col(del_df.columns, ["bowling_team"], required=False)
            if b1: entities.update(del_df[b1].dropna().map(self._norm).tolist())
            if b2: entities.update(del_df[b2].dropna().map(self._norm).tolist())
            
        entities = [e for e in entities if e]
        if not entities: return {}

        ignore = {"stadium", "cricket", "ground", "association", "park", "international"}
        tokens = {e: set([t for t in e.split() if t not in ignore]) for e in entities}
        
        parent = {e: e for e in entities}
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j: parent[root_i] = root_j

        for i, ent_a in enumerate(entities):
            for ent_b in entities[i+1:]:
                tok_a, tok_b = tokens[ent_a], tokens[ent_b]
                if tok_a and tok_b:
                    sim = len(tok_a & tok_b) / len(tok_a | tok_b)
                    if sim >= 0.5: 
                        union(ent_a, ent_b)
                        
        alias_map = {}
        for e in entities:
            root = find(e)
            if len(e) > len(root): 
                parent[root] = e
                
        for e in entities:
            alias_map[e] = find(e)
            
        return alias_map

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

    def _norm(self, text):
        if pd.isna(text): return ""
        clean = str(text).lower().split("(")[0] 
        clean = re.sub(r"[^a-z0-9]+", " ", clean)
        return clean.strip()
        
    def _find_col(self, columns, targets, required=True):
        clean_cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in columns}
        for t in targets:
            clean_t = re.sub(r"[^a-z0-9]", "", t.lower())
            if clean_t in clean_cols:
                return clean_cols[clean_t]
        if required:
            raise ValueError(f"Missing required col: {targets}")
        return None

# ==============================================================================
# MAIN EXECUTION BLOCK (What Docker actually runs)
# ==============================================================================
if __name__ == "__main__":
    # 1. Initialize your custom model
    model = MyModel()
    
    # 2. Load the test file you need to predict on (from your local folder)
    print("Loading test data...")
    test_df = pd.read_csv("test_file.csv") 
    
    # 3. Generate the predictions
    print("Generating predictions...")
    predictions_df = model.predict(test_df)
    
    # 4. Save the predictions to the exact file we created
    predictions_df.to_csv("submission.csv", index=False)
    print("Success! Predictions successfully saved to submission.csv")