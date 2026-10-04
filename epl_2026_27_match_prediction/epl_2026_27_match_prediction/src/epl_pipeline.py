
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd

TEAM_ALIASES = {
    "Man Utd":"Manchester United","Manchester Utd":"Manchester United","Man United":"Manchester United",
    "Man City":"Manchester City","Newcastle":"Newcastle United","Newcastle Utd":"Newcastle United",
    "Nott'ham Forest":"Nottingham Forest","Nott'm Forest":"Nottingham Forest","Nottingham":"Nottingham Forest",
    "Wolves":"Wolverhampton Wanderers","Brighton":"Brighton & Hove Albion",
    "Bournemouth":"AFC Bournemouth","Leeds":"Leeds United","Leicester":"Leicester City",
    "Norwich":"Norwich City","West Brom":"West Bromwich Albion","Cardiff":"Cardiff City",
    "Swansea":"Swansea City","QPR":"Queens Park Rangers","Sheffield Utd":"Sheffield United",
    "Hull":"Hull City","Stoke":"Stoke City","Ipswich":"Ipswich Town","Coventry":"Coventry City",
    "Tottenham":"Tottenham Hotspur"
}

STAT_PAIRS = {
    "shots": ("HomeShots","AwayShots"),
    "sot": ("HomeShotsOnTarget","AwayShotsOnTarget"),
    "corners": ("HomeCorners","AwayCorners"),
    "fouls": ("HomeFouls","AwayFouls"),
    "yellow": ("HomeYellowCards","AwayYellowCards"),
    "red": ("HomeRedCards","AwayRedCards"),
    "xg": ("xG_Home","xG_Away"),
}

PREMATCH_FEATURES = [
    "HomeTeam","AwayTeam","Referee","Home_Elo","Away_Elo","Elo_Difference",
    "Home_PPG_Last5","Away_PPG_Last5","Home_PPG_Last10","Away_PPG_Last10",
    "Home_WinRate_Last5","Away_WinRate_Last5","Home_GoalsFor_Last5","Away_GoalsFor_Last5",
    "Home_GoalsAgainst_Last5","Away_GoalsAgainst_Last5","Home_GoalDiff_Last5","Away_GoalDiff_Last5",
    "Home_HomePPG_Last5","Away_AwayPPG_Last5",
    "Home_shots_Last5","Away_shots_Last5","Home_sot_Last5","Away_sot_Last5",
    "Home_corners_Last5","Away_corners_Last5","Home_fouls_Last5","Away_fouls_Last5",
    "Home_yellow_Last5","Away_yellow_Last5","Home_red_Last5","Away_red_Last5",
    "Home_xg_Last5","Away_xg_Last5",
    "Referee_Known","Referee_History_Matches","Referee_Avg_TotalFouls","Referee_Avg_TotalYellow",
    "Referee_Avg_TotalRed","Referee_Avg_HomeYellow","Referee_Avg_AwayYellow",
    "Referee_FoulBias","Referee_CardBias","Referee_HomeWinRate","Referee_DrawRate",
    "League_Avg_TotalFouls","League_Avg_TotalYellow","Month","DayOfWeek"
]

LIVE_EXTRA_FEATURES = [
    "HalfTimeHomeGoals","HalfTimeAwayGoals","HT_GoalDifference",
    "HomeShots","AwayShots","Shots_Difference","HomeShotsOnTarget","AwayShotsOnTarget","SOT_Difference",
    "HomeCorners","AwayCorners","Corners_Difference","HomeFouls","AwayFouls","Fouls_Difference",
    "HomeYellowCards","AwayYellowCards","Yellow_Difference","HomeRedCards","AwayRedCards","Red_Difference",
    "xG_Home","xG_Away","xG_Difference","Home_Shot_Accuracy","Away_Shot_Accuracy",
    "Shot_Accuracy_Difference","Total_Shots","Total_SOT","Total_Corners","Total_Fouls","Total_Cards"
]

CAT_FEATURES = ["HomeTeam","AwayTeam","Referee"]

LABEL_ORDER = ["A","D","H"]
LABEL_TO_INT = {"A":0,"D":1,"H":2}

def canonical_team(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    return TEAM_ALIASES.get(value, value)

def safe_mean(values, default=0.0):
    vals = [v for v in values if pd.notna(v)]
    return float(np.mean(vals)) if vals else float(default)

def safe_result(home_goals, away_goals):
    if pd.isna(home_goals) or pd.isna(away_goals):
        return np.nan
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"

def _key(df):
    df = df.copy()
    df["DateKey"] = df["MatchDate"].dt.normalize()
    df["MatchKey"] = (
        df["Season"].astype(str) + "|" + df["DateKey"].astype(str) + "|" +
        df["HomeTeam"].astype(str) + "|" + df["AwayTeam"].astype(str)
    )
    return df

def merge_sources(original_csv, matchstats_csv, book_xlsx):
    """Merge the three supplied sources and preserve unplayed fixture rows."""
    o = pd.read_csv(original_csv)
    s = pd.read_csv(matchstats_csv)
    b = pd.read_excel(book_xlsx)

    o["Season"] = o["Season"].astype(str)
    o["MatchDate"] = pd.to_datetime(o["MatchDate"], errors="coerce")
    o["HomeTeam"] = o["HomeTeam"].map(canonical_team)
    o["AwayTeam"] = o["AwayTeam"].map(canonical_team)
    o = _key(o)

    s["Season"] = s["Season"].astype(str)
    s["MatchDate"] = pd.to_datetime(s["Date"], dayfirst=True, errors="coerce")
    s["HomeTeam"] = s["HomeTeam"].map(canonical_team)
    s["AwayTeam"] = s["AwayTeam"].map(canonical_team)
    s["Referee_stats"] = s["Referee"].astype(str).str.strip().replace({"nan":np.nan})
    s = _key(s)

    season_map = {
        "2022-2023":"2022/23","2023-2024":"2023/24",
        "2024-2025":"2024/25","2025-2026":"2025/26"
    }
    b["MatchDate"] = pd.to_datetime(b["Date"], errors="coerce")
    b["Season"] = b["Season"].map(lambda x: season_map.get(str(x), str(x)))
    b["HomeTeam"] = b["Home"].map(canonical_team)
    b["AwayTeam"] = b["Away"].map(canonical_team)
    b["Book_Referee"] = b["Referee"].astype(str).str.strip().replace({"nan":np.nan})
    for c in ["xG","xG.1","FullTimeHomeGoals","FullTimeAwayGoals","Wk"]:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["Book_xG_Home"] = b["xG"]
    b["Book_xG_Away"] = b["xG.1"]
    b["Book_FTHG"] = b["FullTimeHomeGoals"]
    b["Book_FTAG"] = b["FullTimeAwayGoals"]
    b = _key(b)

    combined = o.merge(
        s[["MatchKey","Referee_stats"]].drop_duplicates("MatchKey"),
        on="MatchKey", how="left"
    )
    combined = combined.merge(
        b[["MatchKey","Book_Referee","Book_xG_Home","Book_xG_Away","Book_FTHG","Book_FTAG","Wk","Time"]]
        .drop_duplicates("MatchKey"),
        on="MatchKey", how="left"
    )
    combined["Referee"] = combined["Referee_stats"].combine_first(combined["Book_Referee"])
    combined["xG_Home"] = combined["Book_xG_Home"]
    combined["xG_Away"] = combined["Book_xG_Away"]

    original_keys = set(o["MatchKey"])
    bo = b[~b["MatchKey"].isin(original_keys)].copy()
    append = pd.DataFrame({
        "Season": bo["Season"].values, "MatchDate": bo["MatchDate"].values,
        "HomeTeam": bo["HomeTeam"].values, "AwayTeam": bo["AwayTeam"].values,
        "FullTimeHomeGoals": bo["Book_FTHG"].values, "FullTimeAwayGoals": bo["Book_FTAG"].values,
        "FullTimeResult": [safe_result(h,a) for h,a in zip(bo["Book_FTHG"],bo["Book_FTAG"])],
        "HalfTimeHomeGoals":np.nan, "HalfTimeAwayGoals":np.nan, "HalfTimeResult":np.nan,
        "HomeShots":np.nan,"AwayShots":np.nan,"HomeShotsOnTarget":np.nan,"AwayShotsOnTarget":np.nan,
        "HomeCorners":np.nan,"AwayCorners":np.nan,"HomeFouls":np.nan,"AwayFouls":np.nan,
        "HomeYellowCards":np.nan,"AwayYellowCards":np.nan,"HomeRedCards":np.nan,"AwayRedCards":np.nan,
        "Referee":bo["Book_Referee"].values, "xG_Home":bo["Book_xG_Home"].values,
        "xG_Away":bo["Book_xG_Away"].values, "Wk":bo["Wk"].values,
        "KickoffTime":bo["Time"].values, "MatchKey":bo["MatchKey"].values
    })
    combined = pd.concat([combined, append], ignore_index=True, sort=False)

    numeric_cols = [
        "FullTimeHomeGoals","FullTimeAwayGoals","HalfTimeHomeGoals","HalfTimeAwayGoals",
        "HomeShots","AwayShots","HomeShotsOnTarget","AwayShotsOnTarget","HomeCorners","AwayCorners",
        "HomeFouls","AwayFouls","HomeYellowCards","AwayYellowCards","HomeRedCards","AwayRedCards",
        "xG_Home","xG_Away","Wk"
    ]
    for c in numeric_cols:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")
    combined["FullTimeResult"] = [
        safe_result(h,a) for h,a in zip(combined["FullTimeHomeGoals"], combined["FullTimeAwayGoals"])
    ]
    combined["GoalDifference"] = combined["FullTimeHomeGoals"] - combined["FullTimeAwayGoals"]
    combined = (
        combined.sort_values(["MatchDate","HomeTeam","AwayTeam"])
        .drop_duplicates("MatchKey", keep="first")
        .reset_index(drop=True)
    )
    return combined

def build_features(data, elo_start=1500.0, home_advantage=55.0, k_factor=20.0):
    """Build chronological historical features before updating team/referee state."""
    d = data.sort_values(["MatchDate","HomeTeam","AwayTeam"]).reset_index(drop=True).copy()
    team_hist = defaultdict(lambda: deque(maxlen=10))
    home_hist = defaultdict(lambda: deque(maxlen=10))
    away_hist = defaultdict(lambda: deque(maxlen=10))
    elo = defaultdict(lambda: elo_start)
    ref = defaultdict(lambda: {"n":0,"f":0.0,"y":0.0,"r":0.0,"hy":0.0,"ay":0.0,"hw":0.0,"dr":0.0})
    league_f = 0.0; league_f_n = 0
    league_y = 0.0; league_y_n = 0
    rows=[]
    for _,r in d.iterrows():
        h,a=r.HomeTeam,r.AwayTeam
        hh=list(team_hist[h]); aa=list(team_hist[a])
        rn=str(r.Referee).strip() if pd.notna(r.Referee) and str(r.Referee).strip() else "Unknown"
        feat={"HomeTeam":h,"AwayTeam":a,"Referee":rn,"Home_Elo":elo[h],"Away_Elo":elo[a],
              "Elo_Difference":elo[h]+home_advantage-elo[a]}
        for prefix,hist in [("Home",hh),("Away",aa)]:
            r5=hist[-5:]; r10=hist[-10:]
            feat[f"{prefix}_PPG_Last5"]=safe_mean([x["pts"] for x in r5])
            feat[f"{prefix}_PPG_Last10"]=safe_mean([x["pts"] for x in r10])
            feat[f"{prefix}_WinRate_Last5"]=safe_mean([x["win"] for x in r5])
            feat[f"{prefix}_GoalsFor_Last5"]=safe_mean([x["gf"] for x in r5])
            feat[f"{prefix}_GoalsAgainst_Last5"]=safe_mean([x["ga"] for x in r5])
            feat[f"{prefix}_GoalDiff_Last5"]=safe_mean([x["gf"]-x["ga"] for x in r5])
            for key in STAT_PAIRS:
                feat[f"{prefix}_{key}_Last5"]=safe_mean([x[key] for x in r5])
        feat["Home_HomePPG_Last5"]=safe_mean([x["pts"] for x in list(home_hist[h])[-5:]])
        feat["Away_AwayPPG_Last5"]=safe_mean([x["pts"] for x in list(away_hist[a])[-5:]])

        rs=ref[rn]
        lf=(league_f/league_f_n) if league_f_n else 0.0
        ly=(league_y/league_y_n) if league_y_n else 0.0
        feat["Referee_Known"]=int(rn!="Unknown")
        feat["Referee_History_Matches"]=rs["n"]
        feat["Referee_Avg_TotalFouls"]=rs["f"]/rs["n"] if rs["n"] else lf
        feat["Referee_Avg_TotalYellow"]=rs["y"]/rs["n"] if rs["n"] else ly
        feat["Referee_Avg_TotalRed"]=rs["r"]/rs["n"] if rs["n"] else 0.0
        feat["Referee_Avg_HomeYellow"]=rs["hy"]/rs["n"] if rs["n"] else ly/2
        feat["Referee_Avg_AwayYellow"]=rs["ay"]/rs["n"] if rs["n"] else ly/2
        feat["Referee_FoulBias"]=feat["Referee_Avg_TotalFouls"]-lf
        feat["Referee_CardBias"]=feat["Referee_Avg_TotalYellow"]-ly
        feat["Referee_HomeWinRate"]=rs["hw"]/rs["n"] if rs["n"] else 0.0
        feat["Referee_DrawRate"]=rs["dr"]/rs["n"] if rs["n"] else 0.0
        feat["League_Avg_TotalFouls"]=lf
        feat["League_Avg_TotalYellow"]=ly
        rows.append(feat)

        hg,ag=r.FullTimeHomeGoals,r.FullTimeAwayGoals
        if pd.notna(hg) and pd.notna(ag):
            hg,ag=int(hg),int(ag)
            if hg>ag: hp,ap,hw,aw,actual=3,0,1,0,1.0
            elif hg<ag: hp,ap,hw,aw,actual=0,3,0,1,0.0
            else: hp,ap,hw,aw,actual=1,1,0,0,0.5
            exp=1/(1+10**((elo[a]-(elo[h]+home_advantage))/400))
            delta=k_factor*(actual-exp); elo[h]+=delta; elo[a]-=delta
            hv={"pts":hp,"win":hw,"gf":hg,"ga":ag}; av={"pts":ap,"win":aw,"gf":ag,"ga":hg}
            for key,(hc,ac) in STAT_PAIRS.items():
                hv[key]=r[hc]; av[key]=r[ac]
            team_hist[h].append(hv); team_hist[a].append(av)
            home_hist[h].append(hv); away_hist[a].append(av)
            if rn!="Unknown" and pd.notna(r.HomeFouls) and pd.notna(r.AwayFouls):
                total_f=float(r.HomeFouls+r.AwayFouls)
                home_y=float(r.HomeYellowCards) if pd.notna(r.HomeYellowCards) else 0.0
                away_y=float(r.AwayYellowCards) if pd.notna(r.AwayYellowCards) else 0.0
                total_y=home_y+away_y
                rr=ref[rn]; rr["n"]+=1; rr["f"]+=total_f; rr["y"]+=total_y
                rr["hy"]+=home_y; rr["ay"]+=away_y
                rr["r"]+=(float(r.HomeRedCards) if pd.notna(r.HomeRedCards) else 0.0)+(float(r.AwayRedCards) if pd.notna(r.AwayRedCards) else 0.0)
                rr["hw"]+=int(r.FullTimeResult=="H"); rr["dr"]+=int(r.FullTimeResult=="D")
                league_f+=total_f; league_f_n+=1; league_y+=total_y; league_y_n+=1

    out=pd.concat([d,pd.DataFrame(rows)],axis=1)
    # Protect against duplicate columns when concatenating same-named identity fields.
    out=out.loc[:,~out.columns.duplicated()].copy()
    out["HT_GoalDifference"]=out["HalfTimeHomeGoals"]-out["HalfTimeAwayGoals"]
    out["Shots_Difference"]=out["HomeShots"]-out["AwayShots"]
    out["SOT_Difference"]=out["HomeShotsOnTarget"]-out["AwayShotsOnTarget"]
    out["Corners_Difference"]=out["HomeCorners"]-out["AwayCorners"]
    out["Fouls_Difference"]=out["HomeFouls"]-out["AwayFouls"]
    out["Yellow_Difference"]=out["HomeYellowCards"]-out["AwayYellowCards"]
    out["Red_Difference"]=out["HomeRedCards"]-out["AwayRedCards"]
    out["xG_Difference"]=out["xG_Home"]-out["xG_Away"]
    out["Home_Shot_Accuracy"]=(out["HomeShotsOnTarget"]/out["HomeShots"].replace(0,np.nan)).fillna(0)
    out["Away_Shot_Accuracy"]=(out["AwayShotsOnTarget"]/out["AwayShots"].replace(0,np.nan)).fillna(0)
    out["Shot_Accuracy_Difference"]=out["Home_Shot_Accuracy"]-out["Away_Shot_Accuracy"]
    out["Total_Shots"]=out["HomeShots"]+out["AwayShots"]
    out["Total_SOT"]=out["HomeShotsOnTarget"]+out["AwayShotsOnTarget"]
    out["Total_Corners"]=out["HomeCorners"]+out["AwayCorners"]
    out["Total_Fouls"]=out["HomeFouls"]+out["AwayFouls"]
    out["Total_Cards"]=out["HomeYellowCards"]+out["AwayYellowCards"]+out["HomeRedCards"]+out["AwayRedCards"]
    out["Month"]=out["MatchDate"].dt.month
    out["DayOfWeek"]=out["MatchDate"].dt.dayofweek
    out["GoalDifference"]=out["FullTimeHomeGoals"]-out["FullTimeAwayGoals"]
    return out

def compute_current_state(data, elo_start=1500.0, home_advantage=55.0, k_factor=20.0):
    """Return post-match team/referee state for future prediction."""
    d=data[data.FullTimeResult.notna()].sort_values(["MatchDate","HomeTeam","AwayTeam"])
    teams=defaultdict(lambda:{"elo":elo_start,"hist":deque(maxlen=10),"home_hist":deque(maxlen=10),"away_hist":deque(maxlen=10)})
    refs=defaultdict(list); league_f=[]; league_y=[]
    for _,r in d.iterrows():
        h,a=r.HomeTeam,r.AwayTeam; hg,ag=int(r.FullTimeHomeGoals),int(r.FullTimeAwayGoals)
        if hg>ag: hp,ap,hw,aw,act=3,0,1,0,1.0
        elif hg<ag: hp,ap,hw,aw,act=0,3,0,1,0.0
        else: hp,ap,hw,aw,act=1,1,0,0,0.5
        exp=1/(1+10**((teams[a]["elo"]-(teams[h]["elo"]+home_advantage))/400))
        delta=k_factor*(act-exp); teams[h]["elo"]+=delta; teams[a]["elo"]-=delta
        hv={"pts":hp,"win":hw,"gf":hg,"ga":ag}; av={"pts":ap,"win":aw,"gf":ag,"ga":hg}
        for key,(hc,ac) in STAT_PAIRS.items():
            hv[key]=r[hc]; av[key]=r[ac]
        teams[h]["hist"].append(hv); teams[a]["hist"].append(av)
        teams[h]["home_hist"].append(hv); teams[a]["away_hist"].append(av)
        rn=str(r.Referee).strip() if pd.notna(r.Referee) and str(r.Referee).strip() not in ("","nan") else None
        if rn and pd.notna(r.HomeFouls) and pd.notna(r.AwayFouls):
            refs[rn].append({
                "fouls":float(r.HomeFouls+r.AwayFouls),
                "yellow":float((r.HomeYellowCards if pd.notna(r.HomeYellowCards) else 0)+(r.AwayYellowCards if pd.notna(r.AwayYellowCards) else 0)),
                "red":float((r.HomeRedCards if pd.notna(r.HomeRedCards) else 0)+(r.AwayRedCards if pd.notna(r.AwayRedCards) else 0)),
                "home_y":float(r.HomeYellowCards if pd.notna(r.HomeYellowCards) else 0),
                "away_y":float(r.AwayYellowCards if pd.notna(r.AwayYellowCards) else 0),
                "home_win":int(r.FullTimeResult=="H"),"draw":int(r.FullTimeResult=="D")
            })
            league_f.append(float(r.HomeFouls+r.AwayFouls))
            if pd.notna(r.HomeYellowCards) and pd.notna(r.AwayYellowCards):
                league_y.append(float(r.HomeYellowCards+r.AwayYellowCards))
    return teams,refs,safe_mean(league_f),safe_mean(league_y)

def future_pre_row(home,away,date,teams,refs,league_foul,league_yellow,referee="Unknown"):
    h=teams[home]; a=teams[away]
    hh=list(h["hist"]); aa=list(a["hist"]); rh=refs.get(referee,[])
    lf,ly=league_foul,league_yellow
    rf=safe_mean([x["fouls"] for x in rh],lf); ry=safe_mean([x["yellow"] for x in rh],ly)
    rr=safe_mean([x["red"] for x in rh],0); rhy=safe_mean([x["home_y"] for x in rh],ly/2); ray=safe_mean([x["away_y"] for x in rh],ly/2)
    row={
        "HomeTeam":home,"AwayTeam":away,"Referee":referee,
        "Home_Elo":h["elo"],"Away_Elo":a["elo"],"Elo_Difference":h["elo"]+55-a["elo"],
        "Home_PPG_Last5":safe_mean([x["pts"] for x in hh[-5:]]),"Away_PPG_Last5":safe_mean([x["pts"] for x in aa[-5:]]),
        "Home_PPG_Last10":safe_mean([x["pts"] for x in hh[-10:]]),"Away_PPG_Last10":safe_mean([x["pts"] for x in aa[-10:]]),
        "Home_WinRate_Last5":safe_mean([x["win"] for x in hh[-5:]]),"Away_WinRate_Last5":safe_mean([x["win"] for x in aa[-5:]]),
        "Home_GoalsFor_Last5":safe_mean([x["gf"] for x in hh[-5:]]),"Away_GoalsFor_Last5":safe_mean([x["gf"] for x in aa[-5:]]),
        "Home_GoalsAgainst_Last5":safe_mean([x["ga"] for x in hh[-5:]]),"Away_GoalsAgainst_Last5":safe_mean([x["ga"] for x in aa[-5:]]),
        "Home_GoalDiff_Last5":safe_mean([x["gf"]-x["ga"] for x in hh[-5:]]),"Away_GoalDiff_Last5":safe_mean([x["gf"]-x["ga"] for x in aa[-5:]]),
        "Home_HomePPG_Last5":safe_mean([x["pts"] for x in list(h["home_hist"])[-5:]]),
        "Away_AwayPPG_Last5":safe_mean([x["pts"] for x in list(a["away_hist"])[-5:]]),
        "Home_shots_Last5":safe_mean([x["shots"] for x in hh[-5:]]),"Away_shots_Last5":safe_mean([x["shots"] for x in aa[-5:]]),
        "Home_sot_Last5":safe_mean([x["sot"] for x in hh[-5:]]),"Away_sot_Last5":safe_mean([x["sot"] for x in aa[-5:]]),
        "Home_corners_Last5":safe_mean([x["corners"] for x in hh[-5:]]),"Away_corners_Last5":safe_mean([x["corners"] for x in aa[-5:]]),
        "Home_fouls_Last5":safe_mean([x["fouls"] for x in hh[-5:]]),"Away_fouls_Last5":safe_mean([x["fouls"] for x in aa[-5:]]),
        "Home_yellow_Last5":safe_mean([x["yellow"] for x in hh[-5:]]),"Away_yellow_Last5":safe_mean([x["yellow"] for x in aa[-5:]]),
        "Home_red_Last5":safe_mean([x["red"] for x in hh[-5:]]),"Away_red_Last5":safe_mean([x["red"] for x in aa[-5:]]),
        "Home_xg_Last5":safe_mean([x["xg"] for x in hh[-5:]]),"Away_xg_Last5":safe_mean([x["xg"] for x in aa[-5:]]),
        "Referee_Known":int(referee!="Unknown"),"Referee_History_Matches":len(rh),
        "Referee_Avg_TotalFouls":rf,"Referee_Avg_TotalYellow":ry,"Referee_Avg_TotalRed":rr,
        "Referee_Avg_HomeYellow":rhy,"Referee_Avg_AwayYellow":ray,
        "Referee_FoulBias":rf-lf,"Referee_CardBias":ry-ly,
        "Referee_HomeWinRate":safe_mean([x["home_win"] for x in rh],0),
        "Referee_DrawRate":safe_mean([x["draw"] for x in rh],0),
        "League_Avg_TotalFouls":lf,"League_Avg_TotalYellow":ly,
        "Month":pd.Timestamp(date).month,"DayOfWeek":pd.Timestamp(date).dayofweek
    }
    return pd.DataFrame([row])[PREMATCH_FEATURES]
