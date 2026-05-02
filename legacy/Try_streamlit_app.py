"""
MindGuard v3 - Suicidal Ideation Detector
"""

import streamlit as st
import pickle, numpy as np, time, re, os, datetime, subprocess, tempfile, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pytesseract
import pandas as pd

st.set_page_config(
    page_title="MindGuard - Suicidal Ideation Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
#MainMenu, footer, header { visibility: hidden; }
.stAlert { display: none !important; }
[data-testid="stHeader"]       { display: none !important; height: 0 !important; }
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
html, body { margin: 0; padding: 0; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
.stApp {
    background: linear-gradient(-45deg, #0d1b2a, #132233, #1a0a2e, #0d2233);
    background-size: 400% 400%;
    animation: gradientShift 22s ease infinite;
    font-family: 'Inter', sans-serif;
}
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
.main .block-container { max-width:100% !important; padding:0.6rem 0.8rem 0.4rem 0.8rem !important; margin:0 !important; }
[data-testid="stTabs"] [role="tablist"] { background:rgba(255,255,255,0.06); border-radius:12px; padding:4px; gap:4px; border:1px solid rgba(255,255,255,0.1); flex-wrap:wrap; }
[data-testid="stTabs"] button[role="tab"] { color:rgba(255,255,255,0.6) !important; border-radius:9px !important; font-size:0.76rem !important; font-weight:600 !important; padding:5px 12px !important; border:none !important; transition:all 0.2s; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background:linear-gradient(135deg,#0d9488,#5eead4) !important; color:#0d1b2a !important; }
[data-testid="stTabs"] button[role="tab"]:hover { color:#fff !important; background:rgba(255,255,255,0.1) !important; }
h1,h2,h3,h4 { color:#fff !important; font-weight:700 !important; }
h2 { font-size:1.05rem !important; margin:0 0 0.3rem !important; }
h3 { font-size:0.88rem !important; margin:0.3rem 0 0.2rem !important; }
p,li { color:rgba(255,255,255,0.88) !important; font-size:0.78rem; line-height:1.5; margin:0.05rem 0; }
strong { color:#fff !important; font-weight:600 !important; }
a { color:#5eead4 !important; }
.app-header { display:flex; align-items:center; gap:0.5rem; margin-bottom:0.2rem; }
.app-header-title { font-size:1.2rem; font-weight:700; color:#fff; }
.app-subtitle { font-size:0.72rem; color:rgba(255,255,255,0.65); margin-bottom:0.4rem; }
.divider { border:none; border-top:1px solid rgba(255,255,255,0.1); margin:0.4rem 0; }
.section-label { font-size:0.7rem; font-weight:700; color:#5eead4; letter-spacing:0.1em; text-transform:uppercase; margin:0.55rem 0 0.25rem; }
.stTextArea label, .stTextInput label { color:#fff !important; font-weight:600 !important; font-size:0.78rem !important; }
.stTextArea textarea, .stTextInput input { background:rgba(0,0,0,0.28) !important; color:#fff !important; border:1.5px solid rgba(255,255,255,0.2) !important; border-radius:12px !important; font-size:0.78rem !important; padding:0.55rem 0.7rem !important; caret-color:#fff !important; }
textarea, input[type="text"], input[type="password"] { color:#fff !important; -webkit-text-fill-color:#fff !important; }
input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus, textarea:-webkit-autofill { -webkit-text-fill-color:#fff !important; -webkit-box-shadow:0 0 0px 1000px rgba(0,0,0,0.5) inset !important; transition:background-color 5000s ease-in-out 0s; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color:rgba(94,234,212,0.7) !important; box-shadow:0 0 14px rgba(94,234,212,0.18) !important; outline:none !important; }
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color:rgba(255,255,255,0.32) !important; font-style:italic; }
[data-testid="stFileUploader"] label { color:#fff !important; font-size:0.78rem !important; font-weight:600 !important; }
[data-testid="stFileUploader"] section { background:rgba(0,0,0,0.2) !important; border:1.5px dashed rgba(255,255,255,0.22) !important; border-radius:12px !important; padding:0.5rem !important; }
[data-testid="stFileUploader"] section p { font-size:0.72rem !important; color:rgba(255,255,255,0.5) !important; }
.stButton > button { background:linear-gradient(135deg,#0d9488,#5eead4) !important; color:#0d1b2a !important; font-weight:700 !important; padding:0 1rem !important; border-radius:50px !important; border:none !important; font-size:0.76rem !important; box-shadow:0 4px 14px rgba(13,148,136,0.3) !important; transition:all 0.25s ease !important; width:100%; height:36px; }
.stButton > button:hover { transform:translateY(-2px) !important; }
[data-testid="stDownloadButton"] > button { background:rgba(255,255,255,0.08) !important; color:#fff !important; border:1px solid rgba(255,255,255,0.18) !important; border-radius:50px !important; font-size:0.72rem !important; height:32px; padding:0 0.8rem !important; }
.result-card { background:rgba(255,255,255,0.07); border-radius:14px; padding:0.65rem 0.8rem; margin:0.3rem 0; border:1px solid rgba(255,255,255,0.13); animation:slideUp 0.35s ease-out; }
@keyframes slideUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
.post-card { background:rgba(255,255,255,0.05); border-radius:10px; padding:0.5rem 0.65rem; margin:0.22rem 0; border-left:4px solid #0d9488; font-size:0.74rem; }
.post-card.high   { border-left-color:#ef4444; }
.post-card.medium { border-left-color:#f59e0b; }
.post-card.low    { border-left-color:#22c55e; }
.resource-card { background:rgba(255,255,255,0.05); border-radius:10px; padding:0.42rem 0.6rem; margin:0.18rem 0; border-left:4px solid #7c3aed; font-size:0.74rem; }
.socio-tag { display:inline-block; background:rgba(13,148,136,0.18); color:#5eead4; border-radius:6px; padding:2px 8px; margin:2px; font-size:0.7rem; border:1px solid rgba(13,148,136,0.28); }
.platform-badge { display:inline-block; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); border-radius:8px; padding:0.25rem 0.55rem; font-size:0.7rem; margin-bottom:0.25rem; }
.stat-row { display:flex; gap:0.28rem; margin-bottom:0.28rem; }
.stat-card { flex:1; background:rgba(255,255,255,0.07); border-radius:10px; padding:0.38rem 0.22rem; text-align:center; border:1px solid rgba(255,255,255,0.12); }
.stat-number { font-size:1.1rem; font-weight:700; color:#5eead4; }
.stat-label  { font-size:0.58rem; color:rgba(255,255,255,0.55); text-transform:uppercase; letter-spacing:0.6px; }
.conf-badge { display:inline-block; padding:0.18rem 0.55rem; border-radius:20px; font-size:0.7rem; font-weight:600; }
.conf-high   { background:linear-gradient(135deg,#0d9488,#5eead4); color:#0d1b2a; }
.conf-medium { background:linear-gradient(135deg,#7c3aed,#a78bfa); color:#fff; }
.conf-low    { background:linear-gradient(135deg,#b45309,#fbbf24); color:#0d1b2a; }
.risk-high { color:#fca5a5 !important; font-weight:700 !important; }
.risk-low  { color:#86efac !important; font-weight:700 !important; }
.stProgress > div > div > div > div { background:linear-gradient(90deg,#0d9488,#5eead4); border-radius:6px; height:7px; }
.stProgress > div > div { background:rgba(255,255,255,0.1); border-radius:6px; }
.stWarning { background:rgba(251,191,36,0.1) !important; color:#fff !important; border-left:3px solid #fbbf24 !important; border-radius:9px !important; padding:0.38rem 0.6rem !important; font-size:0.74rem; }
.stInfo    { background:rgba(13,148,136,0.12) !important;  color:#fff !important; border-left:3px solid #0d9488 !important; border-radius:9px !important; padding:0.38rem 0.6rem !important; font-size:0.74rem; }
.stError   { background:rgba(239,68,68,0.15) !important;   color:#fff !important; border-left:3px solid #f87171 !important; border-radius:9px !important; padding:0.38rem 0.6rem !important; font-size:0.74rem; font-weight:600; }
.stSuccess { background:rgba(13,148,136,0.15) !important;  color:#fff !important; border-left:3px solid #5eead4 !important; border-radius:9px !important; padding:0.38rem 0.6rem !important; font-size:0.74rem; }
.support-pill { background:rgba(255,255,255,0.07); border-radius:10px; padding:0.32rem 0.45rem; margin:0.1rem 0; border:1px solid rgba(255,255,255,0.2); font-size:0.68rem; line-height:1.7; text-align:center; color:#fff !important; }
.support-pill strong { display:block; margin-bottom:0.08rem; color:#5eead4 !important; font-size:0.72rem; }
.support-pill a { color:#5eead4 !important; }
.support-pill em { color:rgba(255,255,255,0.85) !important; }
.remember-card { background:rgba(255,255,255,0.07); border-radius:10px; padding:0.38rem 0.55rem; border:1px solid rgba(255,255,255,0.2); font-size:0.72rem; text-align:center; margin-top:0.28rem; color:#fff !important; }
.remember-card strong { color:#5eead4 !important; }
.remember-card span  { color:rgba(255,255,255,0.9) !important; }
.col-footer { font-size:0.63rem; color:rgba(255,255,255,0.38); text-align:center; border-top:1px solid rgba(255,255,255,0.08); padding-top:0.28rem; margin-top:0.45rem; }
.unified-card { background:rgba(255,255,255,0.06); border-radius:12px; padding:0.6rem 0.75rem; margin:0.3rem 0; border:1px solid rgba(255,255,255,0.12); }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background:rgba(0,0,0,0.28) !important; color:#fff !important; border:1.5px solid rgba(255,255,255,0.2) !important; border-radius:12px !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] span { color:#fff !important; }
[data-baseweb="popover"] ul li { color:#0d1b2a !important; background:#f8f8f8 !important; font-size:0.8rem !important; }
[data-baseweb="popover"] ul li:hover { background:#e2e8f0 !important; color:#0d1b2a !important; }
[data-baseweb="popover"] ul li[aria-selected="true"] { background:#0d9488 !important; color:#fff !important; }
[data-baseweb="menu"] { background:#f8f8f8 !important; border-radius:10px !important; }
[data-baseweb="option"] { color:#0d1b2a !important; background:#f8f8f8 !important; }
[data-baseweb="option"]:hover { background:#e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


SAMPLE_TWEETS = {
    "Positive": "Just got promoted at work! Feeling blessed and grateful for this opportunity.",
    "Negative": "I feel like nobody cares anymore. I am so depressed. What's the point of trying?"
}

STOPWORDS = {
    "a","about","above","after","again","all","am","an","and","any","are","as",
    "at","be","because","been","before","being","below","between","both","but",
    "by","cannot","could","did","do","does","doing","don't","down","during",
    "each","few","for","from","get","got","had","has","have","having","he",
    "her","here","hers","herself","him","himself","his","how","i","if","in",
    "into","is","it","its","itself","me","more","most","my","myself","no",
    "nor","not","of","off","on","once","only","or","other","our","out","over",
    "own","same","she","should","so","some","such","than","that","the","their",
    "theirs","them","then","there","these","they","this","those","through","to",
    "too","under","until","up","very","was","we","were","what","when","where",
    "which","while","who","whom","why","will","with","you","your","yours",
}

SOCIOECONOMIC_KEYWORDS = {
    "Employment": [
        "unemployed","fired","jobless","redundant","laid off","no income",
        "lost my job","lost the job","quit my job","cant find work",
        "rejected from","job rejection","no work","out of work",
        "terminated","resignation","job hunting","no job","between jobs",
        "struggling to find work","cant get hired","application rejected",
    ],
    "Housing": [
        "evicted","homeless","eviction","foreclosure","repossessed",
        "cant pay rent","behind on rent","losing my house","lost my home",
        "no place to live","sleeping rough","couch surfing","shelter",
        "kicked out","thrown out","living on the street","no roof",
        "cant afford rent","about to lose my home","housing crisis",
    ],
    "Financial": [
        "broke","debt","bankrupt","bankruptcy","no money","penniless",
        "cant afford","struggling financially","poverty","poor","destitute",
        "bills","overdue","repossession","bailiff","loan shark","in debt",
        "maxed out","credit card debt","financial crisis","cant make ends meet",
        "running out of money","nothing in my account","overdraft",
    ],
    "Relationships": [
        "divorce","divorced","breakup","broke up","separated","cheated on",
        "alone","abandoned","nobody cares","no one cares","nobody loves me",
        "lost my partner","widowed","widower","grief","bereaved","heartbroken",
        "relationship ended","left me","walked out","abusive relationship",
        "domestic violence","isolation","no friends","lost everyone",
        "nobody understands","feel invisible","completely alone",
    ],
    "Health": [
        "chronic pain","terminal","cancer","diagnosis","incurable","disabled",
        "mental illness","depression","anxiety","bipolar","schizophrenia",
        "addiction","addicted","alcoholic","alcohol abuse","drug abuse",
        "overdose","hospitalized","hospital","surgery","treatment failed",
        "no health insurance","cant afford medication","sick","illness",
        "eating disorder","self harm","self-harm","cutting","suicidal thoughts",
    ],
    "Social & Education": [
        "bullied","bullying","expelled","suspended","failed my exams",
        "dropped out","academic failure","social outcast","no friends",
        "excluded","isolated","discriminated","racism","harassment",
        "abused","victim","trauma","ptsd","refugee","asylum seeker",
    ],
}

RESOURCES = {
    "Kenya": [
        {"name":"Befrienders Kenya","contact":"+254 722 178 177","type":"Crisis line"},
        {"name":"Kenya Red Cross","contact":"1199","type":"Emergency"},
        {"name":"Chiromo Hospital Group","contact":"+254 20 4291000","type":"Mental health"},
        {"name":"Mathare Hospital MH Unit","contact":"+254 20 2012185","type":"Hospital"},
    ],
    "USA (National)": [
        {"name":"988 Suicide & Crisis Lifeline","contact":"Call/text 988","type":"Crisis line"},
        {"name":"Crisis Text Line","contact":"Text HOME to 741741","type":"Text-based"},
        {"name":"NAMI Helpline","contact":"1-800-950-6264","type":"Mental health"},
        {"name":"SAMHSA Helpline","contact":"1-800-662-4357","type":"Substance abuse & mental health"},
        {"name":"Veterans Crisis Line","contact":"Call 988, press 1","type":"Veterans"},
        {"name":"Trevor Project (LGBTQ+ youth)","contact":"1-866-488-7386","type":"Youth crisis"},
    ],
    "UK": [
        {"name":"Samaritans","contact":"116 123","type":"Crisis line"},
        {"name":"PAPYRUS (under 35s)","contact":"0800 068 4141","type":"Youth crisis"},
        {"name":"MIND","contact":"0300 123 3393","type":"Mental health"},
        {"name":"Shout","contact":"Text SHOUT to 85258","type":"Text-based"},
    ],
    "Australia": [
        {"name":"Lifeline","contact":"13 11 14","type":"Crisis line"},
        {"name":"Beyond Blue","contact":"1300 22 4636","type":"Mental health"},
        {"name":"Kids Helpline","contact":"1800 55 1800","type":"Youth (5-25)"},
    ],
    "Canada": [
        {"name":"Talk Suicide Canada","contact":"1-833-456-4566","type":"Crisis line"},
        {"name":"Crisis Text Line CA","contact":"Text HOME to 686868","type":"Text-based"},
        {"name":"Kids Help Phone","contact":"1-800-668-6868","type":"Youth"},
    ],
    "International": [
        {"name":"Find A Helpline","contact":"findahelpline.com","type":"Global directory"},
        {"name":"IASP","contact":"https://www.iasp.info/resources/Crisis_Centres/","type":"Global directory"},
        {"name":"Befrienders Worldwide","contact":"https://www.befrienders.org","type":"Global directory"},
    ],
}

US_STATE_RESOURCES = {
    "Alabama":        [{"name":"Alabama Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"AltaPointe Health","contact":"1-800-530-3727","type":"Mental health"}],
    "Alaska":         [{"name":"Careline Crisis Intervention","contact":"1-877-266-4357","type":"Crisis line"},{"name":"Alaska Mental Health Trust","contact":"907-274-7428","type":"Mental health"}],
    "Arizona":        [{"name":"AZ Crisis Line","contact":"1-800-631-1314","type":"Crisis line"},{"name":"Crisis Response Network","contact":"602-222-9444","type":"Crisis line"}],
    "Arkansas":       [{"name":"AR Crisis Line","contact":"1-888-274-7472","type":"Crisis line"},{"name":"UAMS Psychiatric Research","contact":"501-526-8100","type":"Mental health"}],
    "California":     [{"name":"CA Suicide Prevention Hotline","contact":"1-800-784-2433","type":"Crisis line"},{"name":"Didi Hirsch Mental Health","contact":"800-854-7771","type":"Crisis line"}],
    "Colorado":       [{"name":"CO Crisis Services","contact":"1-844-493-8255","type":"Crisis line"},{"name":"Mental Health Center of Denver","contact":"303-504-6500","type":"Mental health"}],
    "Connecticut":    [{"name":"CT Behavioral Health","contact":"1-800-467-3135","type":"Crisis line"},{"name":"DMHAS Crisis Line","contact":"211","type":"Crisis line"}],
    "Delaware":       [{"name":"DE Crisis Hotline","contact":"1-800-652-2929","type":"Crisis line"},{"name":"Connections Community Support","contact":"302-656-8308","type":"Mental health"}],
    "Florida":        [{"name":"FL Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Crisis Center of Tampa Bay","contact":"813-234-1234","type":"Crisis line"}],
    "Georgia":        [{"name":"GA Crisis & Access Line","contact":"1-800-715-4225","type":"Crisis line"},{"name":"Behavioral Health Link","contact":"800-715-4225","type":"Crisis line"}],
    "Hawaii":         [{"name":"HI Crisis Line","contact":"1-800-753-6879","type":"Crisis line"},{"name":"AMHD Crisis Line","contact":"808-832-3100","type":"Mental health"}],
    "Idaho":          [{"name":"ID Careline","contact":"211","type":"Crisis line"},{"name":"Optum Idaho","contact":"1-855-202-0973","type":"Mental health"}],
    "Illinois":       [{"name":"IL Crisis Line","contact":"1-800-345-9049","type":"Crisis line"},{"name":"NAMI Illinois","contact":"1-800-826-4890","type":"Mental health"}],
    "Indiana":        [{"name":"IN Crisis Line","contact":"1-800-662-3445","type":"Crisis line"},{"name":"LifeLine Indiana","contact":"1-800-273-8255","type":"Crisis line"}],
    "Iowa":           [{"name":"IA Warm Line","contact":"1-800-777-3957","type":"Crisis line"},{"name":"MHDS Crisis Line","contact":"1-855-581-8111","type":"Crisis line"}],
    "Kansas":         [{"name":"KS Crisis Line","contact":"1-888-363-2287","type":"Crisis line"},{"name":"COMCARE Crisis","contact":"316-660-7500","type":"Crisis line"}],
    "Kentucky":       [{"name":"KY Crisis Line","contact":"1-800-221-0446","type":"Crisis line"},{"name":"Communicare","contact":"270-769-1304","type":"Mental health"}],
    "Louisiana":      [{"name":"LA Crisis Line","contact":"1-800-259-0570","type":"Crisis line"},{"name":"NAMI Louisiana","contact":"504-835-7633","type":"Mental health"}],
    "Maine":          [{"name":"ME Crisis Line","contact":"1-888-568-1112","type":"Crisis line"},{"name":"NAMI Maine","contact":"1-800-464-5767","type":"Mental health"}],
    "Maryland":       [{"name":"MD Crisis Hotline","contact":"1-800-422-0009","type":"Crisis line"},{"name":"Crisis Link","contact":"703-527-4077","type":"Crisis line"}],
    "Massachusetts":  [{"name":"MA Samaritans","contact":"1-877-870-4673","type":"Crisis line"},{"name":"NAMI Massachusetts","contact":"800-370-9085","type":"Mental health"}],
    "Michigan":       [{"name":"MI Crisis Text Line","contact":"Text HOME to 741741","type":"Text-based"},{"name":"NAMI Michigan","contact":"517-485-4049","type":"Mental health"}],
    "Minnesota":      [{"name":"MN Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Canvas Health Crisis","contact":"651-777-5222","type":"Crisis line"}],
    "Mississippi":    [{"name":"MS Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"NAMI Mississippi","contact":"601-899-9227","type":"Mental health"}],
    "Missouri":       [{"name":"MO Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Places for People","contact":"314-622-4600","type":"Mental health"}],
    "Montana":        [{"name":"MT Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"AWARE Inc","contact":"406-443-1010","type":"Mental health"}],
    "Nebraska":       [{"name":"NE Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Heartland Family Service","contact":"402-553-3000","type":"Mental health"}],
    "Nevada":         [{"name":"NV Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Crisis Support Services of NV","contact":"775-784-8090","type":"Crisis line"}],
    "New Hampshire":  [{"name":"NH Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"NAMI New Hampshire","contact":"1-800-242-6264","type":"Mental health"}],
    "New Jersey":     [{"name":"NJ Hopeline","contact":"1-855-654-6735","type":"Crisis line"},{"name":"NJ Mental Health Cares","contact":"1-866-202-4357","type":"Mental health"}],
    "New Mexico":     [{"name":"NM Crisis Line","contact":"1-855-662-7474","type":"Crisis line"},{"name":"NAMI New Mexico","contact":"505-260-0154","type":"Mental health"}],
    "New York":       [{"name":"NY OMH Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"NYC Well","contact":"1-888-692-9355","type":"Crisis line"},{"name":"NAMI NYC","contact":"212-684-3264","type":"Mental health"}],
    "North Carolina": [{"name":"NC Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Trillium Health Resources","contact":"1-877-685-2415","type":"Mental health"}],
    "North Dakota":   [{"name":"ND Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"FirstStep Recovery","contact":"701-255-3692","type":"Mental health"}],
    "Ohio":           [{"name":"OH Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Crisis Intervention Center","contact":"614-276-2273","type":"Crisis line"}],
    "Oklahoma":       [{"name":"OK Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"NAMI Oklahoma","contact":"405-230-1900","type":"Mental health"}],
    "Oregon":         [{"name":"OR Lines for Life","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Oregon Crisis Network","contact":"503-652-4100","type":"Crisis line"}],
    "Pennsylvania":   [{"name":"PA Crisis Line","contact":"1-855-284-2494","type":"Crisis line"},{"name":"NAMI Pennsylvania","contact":"1-800-223-0500","type":"Mental health"}],
    "Rhode Island":   [{"name":"RI Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Gateway Healthcare","contact":"401-724-8400","type":"Mental health"}],
    "South Carolina": [{"name":"SC Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"NAMI South Carolina","contact":"803-733-9592","type":"Mental health"}],
    "South Dakota":   [{"name":"SD Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Volunteers of America Dakotas","contact":"605-339-1783","type":"Mental health"}],
    "Tennessee":      [{"name":"TN Crisis Line","contact":"1-855-274-7471","type":"Crisis line"},{"name":"NAMI Tennessee","contact":"615-361-6608","type":"Mental health"}],
    "Texas":          [{"name":"TX Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Texas 211","contact":"211","type":"Local resources"},{"name":"NAMI Texas","contact":"512-693-2000","type":"Mental health"}],
    "Utah":           [{"name":"UT Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Utah Crisis Services","contact":"801-587-3000","type":"Crisis line"}],
    "Vermont":        [{"name":"VT Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Howard Center","contact":"802-488-6000","type":"Mental health"}],
    "Virginia":       [{"name":"VA Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"NAMI Virginia","contact":"888-486-8264","type":"Mental health"}],
    "Washington":     [{"name":"WA Crisis Line","contact":"1-866-427-4747","type":"Crisis line"},{"name":"Crisis Connections","contact":"866-427-4747","type":"Crisis line"}],
    "West Virginia":  [{"name":"WV Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"NAMI West Virginia","contact":"304-342-0497","type":"Mental health"}],
    "Wisconsin":      [{"name":"WI Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"Journey Mental Health","contact":"608-280-2600","type":"Mental health"}],
    "Wyoming":        [{"name":"WY Crisis Line","contact":"1-800-273-8255","type":"Crisis line"},{"name":"NAMI Wyoming","contact":"307-432-0837","type":"Mental health"}],
    "Washington D.C.":[{"name":"DC Crisis Line","contact":"1-888-793-4357","type":"Crisis line"},{"name":"DBH Access Helpline","contact":"888-793-4357","type":"Mental health"}],
}

THREE_MONTHS_AGO = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=90)
SIX_MONTHS_AGO  = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=182)

defaults = {
    "analytics":       {"total_analyses":0,"positive_count":0,"negative_count":0,"history":[]},
    "user_input":      "",
    "should_analyze":  False,
    "last_result":     None,
    "input_mode":      "text",
    "download_text":   "",
    "reddit_results":  None,
    "video_result":    None,
    "bluesky_results": None,
    "mastodon_results":None,
    "youtube_results": None,
    "file_results":    None,
    "unified_results": None,
    "bsky_run":        False,
    "bsky_target":     "",
    "bsky_min_run":    0.0,
    "bsky_n_run":      20,
    "bsky_pending":    None,
    "facebook_results": None,
    "twitter_results":  None,
    "fb_pending":       None,
    "tw_pending":       None,
    "fb_triggered":     False,
    "tw_triggered":     False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

for entry in st.session_state.analytics.get("history", []):
    for field, default in [("cls","Unknown"),("ts",""),("prob",0.0),("txt","")]:
        entry.setdefault(field, default)

@st.cache_resource
def load_model_and_tokenizer():
    # Load Mental-RoBERTa — the best performing model (ROC-AUC 0.9813, Accuracy 92.5%)
    # Files needed in the same folder as this script:
    #   mindguard_best_weights.pt  — trained model weights
    #   mindguard_tokenizer/       — folder with tokenizer.json and tokenizer_config.json
    #   mindguard_model_config.json — model configuration
    try:
        with open("mindguard_model_config.json") as f:
            config = json.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("mindguard_tokenizer")
        local_path = "mindguard_model_local"
        load_from  = local_path if os.path.exists(local_path) else config["model_name"]
        model = AutoModelForSequenceClassification.from_pretrained(
            load_from,
            num_labels=2,
            ignore_mismatched_sizes=True,)
        state_dict = torch.load("mindguard_best_weights.pt", map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model, tokenizer, config, device
    except Exception as e:
        st.error(f"Could not load Mental-RoBERTa model: {e}")
        st.stop()

model, tokenizer, model_config, device = load_model_and_tokenizer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)

def predict_one(text: str):
    # Tokenize the raw text — Mental-RoBERTa handles its own preprocessing
    enc = tokenizer(
        text,
        max_length=model_config["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    t0 = time.time()
    with torch.no_grad():
        out   = model(input_ids=enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device))
        probs = torch.softmax(out.logits, dim=1)
        # prob of class 1 = suicidal ideation
        prob  = probs[0][1].item()
    ms = (time.time() - t0) * 1000
    return prob, ms

def predict_batch(texts: list) -> np.ndarray:
    # Batch prediction for multiple texts (used by platform tabs)
    if not texts:
        return np.array([])
    all_probs = []
    batch_size = 16
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=model_config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out   = model(input_ids=enc["input_ids"].to(device),
                          attention_mask=enc["attention_mask"].to(device))
            probs = torch.softmax(out.logits, dim=1)
            # prob of class 1 = suicidal ideation (risk score)
            all_probs.extend(probs[:, 1].cpu().numpy())
    return np.array(all_probs)

def risk_label(score: float):
    if score < 0.35:   return "Low Risk",      "#22c55e", "low"
    elif score < 0.55: return "Moderate Risk", "#f59e0b", "medium"
    elif score < 0.75: return "High Risk",     "#f97316", "high"
    else:              return "Critical Risk", "#ef4444", "high"

def update_analytics(prob, text):
    a = st.session_state.analytics
    a["total_analyses"] += 1
    cls = "Suicidal" if prob >= 0.5 else "Non-Suicidal"
    if prob >= 0.5: a["negative_count"] += 1   # at-risk
    else:           a["positive_count"] += 1   # safe
    a["history"].append({"ts":datetime.datetime.now().strftime("%H:%M"),"cls":cls,"prob":prob,"txt":(text[:38]+"...") if len(text)>38 else text})
    if len(a["history"]) > 10:
        a["history"] = a["history"][-10:]

def run_analysis(text):
    prob, ms = predict_one(text)
    update_analytics(prob, text)
    return prob, ms

def build_download_text(text, prob, ms, source="Text"):
    label = "Suicidal / High Risk" if prob >= 0.5 else "Non-Suicidal / Low Risk"
    risk  = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    conf  = prob if prob >= 0.5 else (1 - prob)
    return (f"Source: {source}\nText:\n{text}\n\nPrediction: {label}\nRisk: {risk}\nConfidence: {conf:.1%}\nLatency: {ms:.1f}ms\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def detect_socioeconomic(posts: list) -> dict:
    result = {}
    for cat, kws in SOCIOECONOMIC_KEYWORDS.items():
        found = []
        for kw in kws:
            kw_clean = re.sub(r"[^a-z0-9\s]", " ", kw.lower()).strip()
            if not kw_clean:
                continue
            for post in posts:
                raw = re.sub(r"['\u2019\u2018`]", "", post["text"].lower())
                raw = re.sub(r"[^a-z0-9\s]", " ", raw)
                raw = re.sub(r"\s+", " ", raw)
                if kw_clean in raw:
                    idx = raw.find(kw_clean)
                    start = max(0, idx - 40)
                    end   = min(len(raw), idx + len(kw_clean) + 40)
                    snippet = "..." + raw[start:end].strip() + "..."
                    found.append({"keyword": kw, "snippet": snippet})
                    break
        result[cat] = found
    return result

def clear_text():
    st.session_state.user_input     = ""
    st.session_state["text_area"]   = ""
    st.session_state.should_analyze = False
    st.session_state.last_result    = None
    st.session_state.download_text  = ""

def extract_text_from_image(image_file):
    try:
        img  = Image.open(image_file).convert("RGB")
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text.strip()
    except Exception:
        return None

def gauge(prob):
    if prob >= 0.5:
        intensity = (prob - 0.5) * 2; clr = "#f87171"; lbl = "Suicidal Risk"
    else:
        intensity = (0.5 - prob) * 2; clr = "#5eead4"; lbl = "Non-Suicidal"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=intensity*100,
        domain={"x":[0,1],"y":[0,1]},
        title={"text":lbl,"font":{"color":"white","size":11}},
        number={"suffix":"%","font":{"color":"white","size":24}},
        gauge={"axis":{"range":[None,100],"tickwidth":1,"tickcolor":"white","tickfont":{"size":8}},"bar":{"color":clr},"bgcolor":"rgba(255,255,255,0.05)","borderwidth":1,"bordercolor":"rgba(255,255,255,0.2)","steps":[{"range":[0,33],"color":"rgba(255,255,255,0.03)"},{"range":[33,66],"color":"rgba(255,255,255,0.06)"},{"range":[66,100],"color":"rgba(255,255,255,0.09)"}],"threshold":{"line":{"color":"white","width":2},"thickness":0.65,"value":80}}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":"white"},height=165,margin=dict(l=6,r=6,t=28,b=2))
    return fig

def timeline_chart(df, date_col="date", score_col="risk_score"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    df["week"] = df[date_col].dt.to_period("W").dt.start_time
    weekly = (df.groupby("week")[score_col].agg(["mean","max","count"]).reset_index().rename(columns={"mean":"avg","max":"peak","count":"posts"}))
    fig = go.Figure()
    for y0,y1,col,lbl in [(0.00,0.35,"rgba(34,197,94,0.07)","Low"),(0.35,0.55,"rgba(245,158,11,0.07)","Moderate"),(0.55,0.75,"rgba(249,115,22,0.07)","High"),(0.75,1.00,"rgba(239,68,68,0.09)","Critical")]:
        fig.add_hrect(y0=y0,y1=y1,fillcolor=col,line_width=0,annotation_text=lbl,annotation_position="right",annotation=dict(font_color="rgba(255,255,255,0.45)",font_size=9))
    fig.add_bar(x=weekly["week"],y=weekly["posts"],name="Posts/week",marker_color="rgba(13,148,136,0.2)",yaxis="y2",hovertemplate="Week %{x}<br>Posts: %{y}<extra></extra>")
    fig.add_scatter(x=weekly["week"],y=weekly["avg"],mode="lines+markers",name="Avg risk",line=dict(color="#5eead4",width=2),marker=dict(size=5,color="#5eead4"),hovertemplate="%{x}<br>Avg: %{y:.1%}<extra></extra>")
    fig.add_scatter(x=weekly["week"],y=weekly["peak"],mode="lines",name="Peak risk",line=dict(color="#ef4444",width=1.5,dash="dot"),hovertemplate="%{x}<br>Peak: %{y:.1%}<extra></extra>")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(19,34,51,0.5)",font_color="rgba(255,255,255,0.7)",yaxis=dict(title="Risk",tickformat=".0%",range=[0,1],gridcolor="rgba(255,255,255,0.07)",color="rgba(255,255,255,0.6)"),yaxis2=dict(overlaying="y",side="right",showgrid=False,color="rgba(255,255,255,0.4)"),xaxis=dict(gridcolor="rgba(255,255,255,0.07)",color="rgba(255,255,255,0.6)"),legend=dict(orientation="h",y=-0.22,font_color="rgba(255,255,255,0.7)",font_size=10),margin=dict(l=40,r=50,t=10,b=40),height=270)
    return fig

def render_post_cards(df, score_col="risk_score", text_col="text", date_col="date", sub_col=None, url_col=None, type_col=None, n=20):
    for _, row in df.sort_values(score_col, ascending=False).head(n).iterrows():
        score = row[score_col]
        lbl, col, cls = risk_label(score)
        preview = str(row[text_col])[:250] + ("..." if len(str(row[text_col])) > 250 else "")
        try:
            date_s = pd.to_datetime(row[date_col]).strftime("%d %b %Y")
        except Exception:
            date_s = str(row.get(date_col,""))
        meta = ""
        if sub_col and sub_col in row: meta += f"r/{row[sub_col]}  "
        if type_col and type_col in row: meta += f"{row[type_col]}  "
        meta += date_s
        link = ""
        if url_col and url_col in row and row[url_col]:
            link = f'<a href="{row[url_col]}" target="_blank" style="color:#5eead4;font-size:0.68rem;text-decoration:none">View source</a>'
        st.markdown(f'<div class="post-card {cls}"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px"><span style="color:rgba(255,255,255,0.5);font-size:0.68rem">{meta}</span><span style="color:{col};font-weight:700;font-size:0.76rem">{score:.1%} — {lbl}</span></div><p style="color:rgba(255,255,255,0.85);margin:0;font-size:0.74rem;line-height:1.5">{preview}</p>{link}</div>', unsafe_allow_html=True)

def render_socio(signals):
    any_found = any(len(v) > 0 for v in signals.values())
    if not any_found:
        st.info("No socio-economic distress keywords detected in this content.")
        return
    total = sum(len(v) for v in signals.values())
    st.markdown(f'<p style="font-size:0.78rem;color:rgba(255,255,255,0.7);margin-bottom:0.5rem"><strong style="color:#5eead4">{total}</strong> socio-economic distress signal(s) detected across <strong style="color:#5eead4">{sum(1 for v in signals.values() if v)}</strong> categories.</p>', unsafe_allow_html=True)
    for cat, items in signals.items():
        if not items:
            continue
        st.markdown(f'<p style="font-size:0.78rem;font-weight:700;margin:0.6rem 0 0.15rem;color:#fff">{cat} <span style="color:#5eead4;font-weight:400;font-size:0.72rem">({len(items)} signal(s))</span></p>', unsafe_allow_html=True)
        for item in items:
            kw      = item["keyword"] if isinstance(item, dict) else item
            snippet = item.get("snippet","") if isinstance(item, dict) else ""
            st.markdown(f'<div style="background:rgba(13,148,136,0.12);border-radius:8px;padding:0.4rem 0.65rem;margin:0.18rem 0;border-left:3px solid #0d9488;"><span style="color:#5eead4;font-weight:700;font-size:0.76rem">{kw}</span>' + (f'<br><span style="color:rgba(255,255,255,0.55);font-size:0.7rem;font-style:italic">{snippet}</span>' if snippet else "") + '</div>', unsafe_allow_html=True)
    found_cats = {c: len(v) for c, v in signals.items() if v}
    if len(found_cats) > 1:
        fig = px.pie(names=list(found_cats.keys()),values=list(found_cats.values()),hole=0.45,color_discrete_sequence=["#0d9488","#7c3aed","#f97316","#f59e0b","#ef4444","#22c55e"])
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="rgba(255,255,255,0.8)",margin=dict(t=10,b=10,l=10,r=10),height=260,showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def format_contact(contact: str) -> str:
    if contact.startswith("http://") or contact.startswith("https://"):
        return f'<a href="{contact}" target="_blank" style="color:#5eead4;font-weight:600;font-size:0.78rem;text-decoration:underline">{contact}</a>'
    if contact.endswith(".com") or contact.endswith(".org") or contact.endswith(".info") or contact.endswith(".net"):
        return f'<a href="https://{contact}" target="_blank" style="color:#5eead4;font-weight:600;font-size:0.78rem;text-decoration:underline">{contact}</a>'
    return f'<span style="color:#5eead4;font-weight:600;font-size:0.78rem">{contact}</span>'

def render_resource_card(r: dict, border_color: str = "#7c3aed"):
    contact_html = format_contact(r["contact"])
    st.markdown(f'<div class="resource-card" style="border-left-color:{border_color}"><div style="font-weight:700;color:#fff;font-size:0.8rem">{r["name"]}</div><div style="color:rgba(255,255,255,0.45);font-size:0.68rem;margin:1px 0">{r["type"]}</div><div>{contact_html}</div></div>', unsafe_allow_html=True)

def render_resources(region):
    for r in RESOURCES[region]:
        render_resource_card(r)

def overall_banner(score, n_posts, n_high, period):
    lbl, col, _ = risk_label(score)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Overall Risk", f"{score:.1%}")
    m2.metric("Posts Analysed", str(n_posts))
    m3.metric("High-Risk Posts", str(n_high))
    m4.metric("Period", period)
    st.markdown(f'<div style="display:inline-block;padding:5px 16px;border-radius:999px;background:{col}22;color:{col};border:1.5px solid {col};font-weight:700;font-size:0.88rem;margin:4px 0">{lbl}</div>', unsafe_allow_html=True)
    if score >= 0.55:
        st.error("CRISIS ALERT — High-risk content detected. Please direct to crisis resources.")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_reddit(username, client_id, client_secret):
    import praw
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent="MindGuard:v3.0 (mental health research)")
    posts = []
    try:
        redditor = reddit.redditor(username)
        for sub in redditor.submissions.new(limit=200):
            dt = datetime.datetime.fromtimestamp(sub.created_utc, tz=datetime.timezone.utc)
            if dt < SIX_MONTHS_AGO: break
            text = f"{sub.title} {sub.selftext}".strip()
            if len(text) > 10:
                posts.append({"text":text,"date":dt,"subreddit":str(sub.subreddit),"type":"post","url":f"https://reddit.com{sub.permalink}"})
        for c in redditor.comments.new(limit=500):
            dt = datetime.datetime.fromtimestamp(c.created_utc, tz=datetime.timezone.utc)
            if dt < SIX_MONTHS_AGO: break
            text = c.body.strip()
            if len(text) > 10 and text not in ("[deleted]","[removed]"):
                posts.append({"text":text,"date":dt,"subreddit":str(c.subreddit),"type":"comment","url":f"https://reddit.com{c.permalink}"})
    except Exception as e:
        raise RuntimeError(str(e))
    posts.sort(key=lambda x: x["date"])
    return posts

def download_audio(url, out_dir):
    """Download audio and trim to first 5 minutes."""
    out_template = os.path.join(out_dir, "audio.%(ext)s")
    cmd = ["yt-dlp","--extract-audio","--audio-format","mp3","--audio-quality","5","--no-playlist","--quiet","-o",out_template,url]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr.strip()}")
    mp3 = os.path.join(out_dir, "audio.mp3")
    if not os.path.exists(mp3):
        files = list(Path(out_dir).glob("audio.*"))
        if not files:
            raise RuntimeError("Audio file not found after download.")
        mp3 = str(files[0])
    trimmed = os.path.join(out_dir, "audio_trimmed.mp3")
    trim_cmd = ["ffmpeg","-i",mp3,"-t","300","-acodec","libmp3lame","-q:a","5","-y",trimmed,"-loglevel","error"]
    try:
        trim_result = subprocess.run(trim_cmd, capture_output=True, timeout=60)
        if trim_result.returncode == 0 and os.path.exists(trimmed) and os.path.getsize(trimmed) > 1000:
            return trimmed
    except Exception:
        pass
    return mp3

def transcribe_audio(audio_path):
    from faster_whisper import WhisperModel
    wm = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = wm.transcribe(audio_path, beam_size=3)
    return " ".join(seg.text.strip() for seg in segments).strip()

def bluesky_login(identifier: str, password: str) -> str:
    import urllib.request
    url     = "https://bsky.social/xrpc/com.atproto.server.createSession"
    payload = json.dumps({"identifier": identifier, "password": password}).encode()
    req     = urllib.request.Request(url, data=payload, headers={"Content-Type":"application/json","User-Agent":"MindGuard/3.0"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
            return data["accessJwt"]
    except Exception as e:
        raise RuntimeError(f"Bluesky login failed: {e}")

def fetch_bluesky(handle: str, access_token: str = None) -> list:
    import urllib.request, urllib.error
    handle = handle.strip().lstrip("@")
    if "." not in handle:
        handle = f"{handle}.bsky.social"
    cutoff  = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=90)
    headers = {"User-Agent": "MindGuard/3.0"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    resolve_url = f"https://bsky.social/xrpc/com.atproto.identity.resolveHandle?handle={handle}"
    try:
        req = urllib.request.Request(resolve_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            did = json.loads(r.read().decode())["did"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Handle not found: {handle} (HTTP {e.code}).")
    except Exception as e:
        raise RuntimeError(f"Could not reach Bluesky API: {e}")
    posts = []; cursor = None
    for _ in range(10):
        params = f"actor={did}&limit=100"
        if cursor: params += f"&cursor={cursor}"
        feed_url = f"https://bsky.social/xrpc/app.bsky.feed.getAuthorFeed?{params}"
        try:
            req = urllib.request.Request(feed_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Could not fetch posts (HTTP {e.code}). Please provide Bluesky credentials.")
        except Exception as e:
            raise RuntimeError(f"Could not fetch posts: {e}")
        feed = data.get("feed", [])
        if not feed: break
        oldest_in_page = None
        for item in feed:
            post    = item.get("post", {})
            record  = post.get("record", {})
            created = record.get("createdAt", "")
            try:
                dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:
                continue
            oldest_in_page = dt
            if dt < cutoff: continue
            text = record.get("text", "").strip()
            if len(text) > 5:
                uri  = post.get("uri", "")
                rkey = uri.split("/")[-1] if uri else ""
                posts.append({"text":text,"date":dt,"url":f"https://bsky.app/profile/{handle}/post/{rkey}"})
        cursor = data.get("cursor")
        if not cursor: break
        if oldest_in_page and oldest_in_page < cutoff: break
    posts.sort(key=lambda x: x["date"])
    return posts

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_mastodon(handle):
    import urllib.request
    if "@" not in handle:
        raise RuntimeError("Mastodon handle must be in format: username@instance.social")
    parts = handle.lstrip("@").split("@")
    if len(parts) != 2:
        raise RuntimeError("Mastodon handle must be in format: username@instance.social")
    username, instance = parts
    search_url = f"https://{instance}/api/v1/accounts/lookup?acct={username}"
    try:
        with urllib.request.urlopen(search_url, timeout=10) as r:
            account = json.loads(r.read())
            account_id = account["id"]
    except Exception as e:
        raise RuntimeError(f"Could not find Mastodon account: {e}")
    posts = []; max_id = None
    for _ in range(10):
        params = f"limit=40&exclude_replies=false"
        if max_id: params += f"&max_id={max_id}"
        status_url = f"https://{instance}/api/v1/accounts/{account_id}/statuses?{params}"
        try:
            with urllib.request.urlopen(status_url, timeout=10) as r:
                statuses = json.loads(r.read())
        except Exception:
            break
        if not statuses: break
        for s in statuses:
            created = s.get("created_at","")
            try:
                dt = datetime.datetime.fromisoformat(created.replace("Z","+00:00"))
            except Exception:
                continue
            if dt < THREE_MONTHS_AGO: break
            content = re.sub(r"<[^>]+>","",s.get("content","")).strip()
            if len(content) > 5:
                posts.append({"text":content,"date":dt,"url":s.get("url","")})
        max_id = statuses[-1]["id"]
        if posts and posts[-1]["date"] < THREE_MONTHS_AGO: break
    posts = [p for p in posts if p["date"] >= THREE_MONTHS_AGO]
    posts.sort(key=lambda x: x["date"])
    return posts

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_youtube(channel_input, api_key):
    import urllib.request, urllib.parse
    BASE = "https://www.googleapis.com/youtube/v3"
    def yt_get(endpoint, params):
        params["key"] = api_key
        url = f"{BASE}/{endpoint}?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    channel_id = None
    if "youtube.com/channel/" in channel_input:
        channel_id = channel_input.split("youtube.com/channel/")[-1].split("/")[0].split("?")[0]
    elif "youtube.com/@" in channel_input:
        handle = channel_input.split("youtube.com/@")[-1].split("/")[0].split("?")[0]
        data = yt_get("channels", {"part":"id","forHandle":handle})
        items = data.get("items",[])
        if items: channel_id = items[0]["id"]
    else:
        data = yt_get("channels", {"part":"id","forHandle":channel_input.lstrip("@")})
        items = data.get("items",[])
        if items: channel_id = items[0]["id"]
    if not channel_id:
        raise RuntimeError("Could not resolve YouTube channel. Use a channel URL or @handle.")
    cutoff = THREE_MONTHS_AGO.strftime("%Y-%m-%dT%H:%M:%SZ")
    search_data = yt_get("search", {"part":"id,snippet","channelId":channel_id,"type":"video","order":"date","maxResults":50,"publishedAfter":cutoff})
    posts = []
    for item in search_data.get("items",[]):
        video_id  = item["id"].get("videoId","")
        snippet   = item.get("snippet",{})
        title     = snippet.get("title","")
        desc      = snippet.get("description","")
        published = snippet.get("publishedAt","")
        try:
            dt = datetime.datetime.fromisoformat(published.replace("Z","+00:00"))
        except Exception:
            continue
        text = f"{title} {desc}".strip()
        if len(text) > 5:
            posts.append({"text":text,"date":dt,"url":f"https://youtube.com/watch?v={video_id}","video_id":video_id})
        try:
            comments_data = yt_get("commentThreads", {"part":"snippet","videoId":video_id,"maxResults":20,"order":"relevance"})
            for c in comments_data.get("items",[]):
                comment_text = c["snippet"]["topLevelComment"]["snippet"].get("textDisplay","")
                comment_text = re.sub(r"<[^>]+>","",comment_text).strip()
                if len(comment_text) > 5:
                    posts.append({"text":comment_text,"date":dt,"url":f"https://youtube.com/watch?v={video_id}","video_id":video_id})
        except Exception:
            pass
    posts.sort(key=lambda x: x["date"])
    return posts

def parse_whatsapp_line(line: str):
    m = re.match(r"\[(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4}),\s*(\d{1,2}):(\d{2})(?::\d{2})?\]\s*([^:]+):\s*(.+)", line)
    if m:
        day,month,year,hour,minute,sender,msg = m.groups()
        year = int(year); year = year+2000 if year < 100 else year
        try:
            dt = datetime.datetime(year,int(month),int(day),int(hour),int(minute),tzinfo=datetime.timezone.utc)
            return dt, sender.strip(), msg.strip()
        except Exception:
            pass
    m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4}),\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*-\s*([^:]+):\s*(.+)", line)
    if m:
        day,month,year,hour,minute,sender,msg = m.groups()
        year = int(year); year = year+2000 if year < 100 else year
        try:
            dt = datetime.datetime(year,int(month),int(day),int(hour),int(minute),tzinfo=datetime.timezone.utc)
            return dt, sender.strip(), msg.strip()
        except Exception:
            pass
    return None

def parse_uploaded_file(uploaded_file):
    posts = []; name = uploaded_file.name.lower(); content = uploaded_file.read()
    if name.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore"); lines = text.split("\n")
        whatsapp_hits = sum(1 for l in lines[:20] if re.search(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}.*\d{1,2}:\d{2}", l))
        is_whatsapp = whatsapp_hits >= 2
        if is_whatsapp:
            skip_phrases = ["messages and calls are end-to-end encrypted","message was deleted","you deleted this message","missed voice call","missed video call","image omitted","video omitted","audio omitted","document omitted","sticker omitted","gif omitted","contact card omitted","location omitted","this message was deleted","changed the subject","changed this group","added you","left","joined using this group"]
            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                parsed = parse_whatsapp_line(line)
                if parsed:
                    dt, sender, msg = parsed
                    if any(skip in msg.lower() for skip in skip_phrases): continue
                    if len(msg) < 3: continue
                    posts.append({"text":msg,"date":dt,"url":"","sender":sender})
                else:
                    if posts and len(line) > 3:
                        posts[-1]["text"] += " " + line
        else:
            valid_lines = [l.strip() for l in lines if len(l.strip()) > 10]
            for i, line in enumerate(valid_lines):
                dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=i)
                posts.append({"text":line,"date":dt,"url":""})
    elif name.endswith(".csv"):
        import io; df = pd.read_csv(io.BytesIO(content))
        text_col = next((c for c in df.columns if any(k in c.lower() for k in ["text","content","message","post","body","tweet","comment"])), None)
        if text_col is None and len(df.columns) > 0: text_col = df.columns[0]
        date_col = next((c for c in df.columns if any(k in c.lower() for k in ["date","time","created","timestamp"])), None)
        for i, row in df.iterrows():
            text = str(row[text_col]).strip() if text_col else ""
            if len(text) < 5: continue
            dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=i)
            if date_col:
                try: dt = pd.to_datetime(row[date_col], utc=True)
                except Exception: pass
            posts.append({"text":text,"date":dt,"url":""})
    elif name.endswith(".json"):
        try:
            data = json.loads(content.decode("utf-8", errors="ignore"))
        except Exception:
            return posts
        if isinstance(data, list):
            for item in data:
                if "timestamp" in item and "data" in item:
                    ts = item.get("timestamp", 0); dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                    text = ""
                    for d in item.get("data", []):
                        if isinstance(d, dict):
                            text += d.get("post", {}).get("message","") if isinstance(d.get("post"),dict) else str(d.get("post",""))
                    if not text: text = item.get("title","")
                    if len(text.strip()) > 5: posts.append({"text":text.strip(),"date":dt,"url":""})
                elif "tweet" in item:
                    tweet = item["tweet"]; text = tweet.get("full_text", tweet.get("text","")).strip()
                    created = tweet.get("created_at",""); dt = datetime.datetime.now(datetime.timezone.utc)
                    try: dt = datetime.datetime.strptime(created, "%a %b %d %H:%M:%S +0000 %Y").replace(tzinfo=datetime.timezone.utc)
                    except Exception: pass
                    if len(text) > 5 and not text.startswith("RT @"): posts.append({"text":text,"date":dt,"url":f"https://twitter.com/i/web/status/{tweet.get('id_str','')}"})
        elif isinstance(data, dict):
            tweets = data.get("tweets", data.get("data", []))
            if isinstance(tweets, list):
                for item in tweets:
                    tweet = item.get("tweet", item); text = tweet.get("full_text", tweet.get("text","")).strip()
                    created = tweet.get("created_at",""); dt = datetime.datetime.now(datetime.timezone.utc)
                    try: dt = datetime.datetime.strptime(created, "%a %b %d %H:%M:%S +0000 %Y").replace(tzinfo=datetime.timezone.utc)
                    except Exception: pass
                    if len(text) > 5 and not text.startswith("RT @"): posts.append({"text":text,"date":dt,"url":""})
    return posts

def _run_scraper_worker(platform: str, url: str, months: int) -> list:
    import sys as _sys
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scraper_worker.py")
    result = subprocess.run([_sys.executable, worker, platform, url, str(months)], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        err = (result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown error")
        raise RuntimeError(err)
    try:
        data = json.loads(result.stdout.strip())
    except Exception:
        raise RuntimeError(f"Could not parse scraper output: {result.stdout[:300]}")
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "Scraper failed"))
    posts = []
    for p in data.get("posts", []):
        try:
            dt = datetime.datetime.fromisoformat(p["date"]).replace(tzinfo=datetime.timezone.utc)
        except Exception:
            dt = datetime.datetime.now(datetime.timezone.utc)
        posts.append({"text":p["text"],"date":dt,"url":p["url"]})
    return posts

def scrape_facebook_public(profile_url: str, months: int = 3) -> list:
    return _run_scraper_worker("facebook", profile_url, months)

def scrape_twitter_public(profile_url: str, months: int = 3) -> list:
    return _run_scraper_worker("twitter", profile_url, months)


st.markdown("""
<div class="app-header">
    <span style="font-size:1.6rem">🧠</span>
    <div>
        <div class="app-header-title">MindGuard</div>
        <div class="app-subtitle">Early detection of suicidal ideation — Mental-RoBERTa NLP model</div>
    </div>
</div>
""", unsafe_allow_html=True)

(tab_text, tab_reddit, tab_video, tab_bluesky,
 tab_mastodon, tab_youtube, tab_file,
 tab_facebook, tab_twitter,
 tab_unified, tab_resources) = st.tabs([
    "Text / Image","Reddit","Video (any platform)","Bluesky",
    "Mastodon","YouTube","File Upload","Facebook","Twitter / X",
    "Multi-Platform","Resources",
])

with tab_text:
    colA, colB, colC = st.columns([1.0, 1.25, 1.05])
    with colA:
        st.markdown('<h2>Input</h2>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            if st.button("Type Text", use_container_width=True):
                st.session_state.input_mode = "text"; st.rerun()
        with m2:
            if st.button("Upload Image", use_container_width=True):
                st.session_state.input_mode = "image"; st.rerun()
        st.markdown('<div style="margin-top:0.3rem"></div>', unsafe_allow_html=True)
        if st.session_state.input_mode == "text":
            with st.expander("Try a sample", expanded=False):
                for label, tweet in SAMPLE_TWEETS.items():
                    if st.button(label, key=f"sample_{label}", use_container_width=True):
                        st.session_state.user_input = tweet; st.session_state["text_area"] = tweet; st.session_state.should_analyze = True; st.rerun()
            user_input = st.text_area("Enter text to analyse:", height=108, placeholder="Type or paste text here...", value=st.session_state.user_input, key="text_area")
            st.session_state.user_input = user_input
            b1, b2 = st.columns([1.6, 1])
            with b1: analyze_btn = st.button("Analyse", use_container_width=True, key="analyze_text")
            with b2: st.button("Clear", use_container_width=True, on_click=clear_text)
            if analyze_btn:
                if user_input.strip():
                    p, ms = run_analysis(user_input)
                    st.session_state.last_result   = {"prob":p,"ms":ms,"text":user_input,"ok":True}
                    st.session_state.download_text = build_download_text(user_input,p,ms)
                else:
                    st.session_state.last_result = {"ok":False,"empty":True}
                st.rerun()
            if st.session_state.should_analyze and st.session_state.user_input.strip():
                st.session_state.should_analyze = False
                p, ms = run_analysis(st.session_state.user_input)
                st.session_state.last_result   = {"prob":p,"ms":ms,"text":st.session_state.user_input,"ok":True}
                st.session_state.download_text = build_download_text(st.session_state.user_input,p,ms)
                st.rerun()
        else:
            uploaded_file = st.file_uploader("Upload a screenshot:", type=["png","jpg","jpeg","webp"], label_visibility="collapsed")
            if uploaded_file:
                st.image(Image.open(uploaded_file), use_container_width=True, caption="Uploaded")
            ib1, ib2 = st.columns([1.6, 1])
            with ib1: analyze_img_btn = st.button("Analyse Image", use_container_width=True, key="analyze_image")
            with ib2: st.button("Clear", use_container_width=True, on_click=clear_text, key="clear_image")
            if analyze_img_btn:
                if uploaded_file:
                    with st.spinner("Reading text from image..."):
                        extracted = extract_text_from_image(uploaded_file)
                    if extracted:
                        p, ms = run_analysis(extracted)
                        st.session_state.last_result   = {"prob":p,"ms":ms,"text":extracted,"ok":True,"from_image":True}
                        st.session_state.download_text = build_download_text(extracted,p,ms,"Image OCR")
                    else:
                        st.session_state.last_result = {"ok":False,"ocr_fail":True}
                else:
                    st.session_state.last_result = {"ok":False,"no_image":True}
                st.rerun()
        st.markdown('<div class="col-footer">MindGuard v3 — Mental-RoBERTa — Mental Health Research</div>', unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <p class="section-label">Crisis Helplines — 24/7</p>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.18rem;margin-bottom:0.3rem">
            <div class="support-pill"><strong>Kenya</strong><span style="color:#fff">Befrienders</span><br><span style="color:#5eead4;font-weight:600">+254 722 178 177</span></div>
            <div class="support-pill"><strong>USA</strong><span style="color:#fff">988 Lifeline</span><br><span style="color:#5eead4;font-weight:600">988</span><br><span style="color:#fff">Text HOME to </span><span style="color:#5eead4;font-weight:600">741741</span></div>
            <div class="support-pill"><strong>UK</strong><span style="color:#fff">Samaritans</span><br><span style="color:#5eead4;font-weight:600">116 123</span></div>
            <div class="support-pill"><strong>Global</strong><a href="https://findahelpline.com" target="_blank" style="color:#5eead4">findahelpline.com</a></div>
        </div>
        <hr class="divider">
        """, unsafe_allow_html=True)
        r = st.session_state.last_result
        if r and not r.get("ok"):
            if r.get("empty"):      st.warning("Please enter some text first.")
            elif r.get("no_image"): st.warning("Please upload an image first.")
            elif r.get("ocr_fail"): st.warning("Could not read text from the image.")
        if r and r.get("ok"):
            prob = r["prob"]
            label    = "Suicidal / Negative"   if prob < 0.5 else "Non-Suicidal / Positive"
            color    = "#f87171"               if prob >= 0.5 else "#5eead4"
            risk_lbl = "HIGH RISK"             if prob >= 0.5 else "LOW RISK"
            risk_cls = "risk-high"             if prob >= 0.5 else "risk-low"
            conf     = prob if prob >= 0.5 else (1 - prob)
            if conf >= 0.8:   cl,cc = "High Confidence",   "conf-high"
            elif conf >= 0.6: cl,cc = "Medium Confidence", "conf-medium"
            else:             cl,cc = "Low Confidence",    "conf-low"
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:1rem;font-weight:700;color:{color};text-align:center;margin:0 0 0.3rem">{label}</p>', unsafe_allow_html=True)
            st.plotly_chart(gauge(prob), use_container_width=True)
            st.markdown(f'<p style="font-size:0.76rem;margin:0.1rem 0"><strong>Risk:</strong> <span class="{risk_cls}">{risk_lbl}</span></p>', unsafe_allow_html=True)
            st.progress(int(prob*100) if prob >= 0.5 else int((1-prob)*100))
            st.markdown(f'<div style="text-align:center;margin:0.18rem 0"><span class="conf-badge {cc}">{cl}: {conf:.1%}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;margin:0.18rem 0 0.32rem"><span style="background:rgba(13,148,136,0.22);color:#5eead4;padding:3px 11px;border-radius:999px;font-size:0.67rem;font-weight:600">{r["ms"]:.1f}ms</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if prob >= 0.5:
                st.error("CRISIS ALERT — High-risk content detected. Use the helplines above.")
            if st.session_state.download_text:
                st.download_button("Download report", st.session_state.download_text, file_name="mindguard_report.txt", use_container_width=True)
        st.markdown('<div class="remember-card"><strong style="font-size:0.72rem;color:#5eead4;display:block;margin-bottom:0.08rem">Remember</strong><span style="color:#fff">You are not alone</span> <span style="color:rgba(255,255,255,0.4)">&nbsp;·&nbsp;</span> <span style="color:#fff">Help is available 24/7</span> <span style="color:rgba(255,255,255,0.4)">&nbsp;·&nbsp;</span> <span style="color:#fff">Talking helps</span></div>', unsafe_allow_html=True)

    with colC:
        st.markdown('<h3 style="text-align:center;margin:0 0 0.3rem">Session Analytics</h3>', unsafe_allow_html=True)
        a = st.session_state.analytics
        if a["total_analyses"] > 0:
            st.markdown(f'<div class="stat-row"><div class="stat-card"><div class="stat-label">Total</div><div class="stat-number">{a["total_analyses"]}</div></div><div class="stat-card"><div class="stat-label">Positive</div><div class="stat-number" style="color:#5eead4">{a["positive_count"]}</div></div><div class="stat-card"><div class="stat-label">At-Risk</div><div class="stat-number" style="color:#f87171">{a["negative_count"]}</div></div></div>', unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(labels=["Non-Suicidal","Suicidal Risk"],values=[a["positive_count"],a["negative_count"]],marker_colors=["#5eead4","#f87171"],hole=0.38,textfont_size=10,textfont_color="white"))
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":"white"},height=175,margin=dict(l=5,r=5,t=8,b=5),legend=dict(font=dict(color="white",size=9),orientation="v",x=1.0,y=0.5))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.72rem;font-weight:600;margin-bottom:0.12rem">Recent</p>', unsafe_allow_html=True)
            for item in reversed(a["history"][-5:]):
                dot = "+" if item["cls"] == "Positive" else "-"
                st.markdown(f'<p style="margin:0.06rem 0;font-size:0.68rem">[{dot}] <strong>{item["cls"]}</strong> · {item["ts"]} · {item["prob"]:.0%}<br><em style="color:rgba(255,255,255,0.5)">{item["txt"]}</em></p>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;padding:2rem 0.5rem;color:rgba(255,255,255,0.4)"><p style="font-size:0.76rem">No analyses yet.</p></div>', unsafe_allow_html=True)

    r = st.session_state.last_result
    if r and r.get("ok"):
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Socio-Economic Signals Detected</p>', unsafe_allow_html=True)
        render_socio(detect_socioeconomic([{"text": r["text"]}]))


with tab_reddit:
    rA, rB = st.columns([1, 2])
    with rA:
        st.markdown('<h2>Reddit User Analysis</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.6)">Fetches 6 months of posts and comments via the free Reddit API.</p>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        reddit_user = st.text_input("Username (without u/)", placeholder="e.g. spez", key="reddit_username")
        with st.expander("Reddit API credentials", expanded=True):
            st.markdown('<p style="font-size:0.7rem">Free at <a href="https://www.reddit.com/prefs/apps" target="_blank">reddit.com/prefs/apps</a> — create a script app.</p>', unsafe_allow_html=True)
            r_id     = st.text_input("Client ID",     value=os.getenv("REDDIT_CLIENT_ID",""),     placeholder="under app name")
            r_secret = st.text_input("Client Secret", value=os.getenv("REDDIT_CLIENT_SECRET",""), type="password")
        min_risk = st.slider("Show posts above risk score", 0.0, 1.0, 0.0, 0.05, key="r_min")
        n_show   = st.slider("Max posts to display", 5, 50, 20, 5, key="r_n")
        fetch_btn = st.button("Analyse Reddit User", use_container_width=True, key="reddit_fetch")
        if fetch_btn:
            if not reddit_user.strip(): st.warning("Enter a username.")
            elif not r_id or not r_secret: st.error("Enter your Reddit API credentials.")
            else:
                uname = reddit_user.strip().lstrip("u/")
                with st.spinner(f"Fetching posts for u/{uname}..."):
                    try:
                        raw = fetch_reddit(uname, r_id, r_secret)
                    except RuntimeError as e:
                        st.error(str(e)); raw = []
                if raw:
                    with st.spinner(f"Running Bi-LSTM on {len(raw)} posts..."):
                        scores = predict_batch([clean_text(p["text"]) for p in raw])
                    df = pd.DataFrame(raw); df["risk_score"] = scores; df["date"] = pd.to_datetime(df["date"], utc=True)
                    st.session_state.reddit_results = {"username":uname,"df":df,"overall":float(np.percentile(scores,85)),"n_high":int((scores>=0.55).sum()),"signals":detect_socioeconomic(raw),"n_posts":len(raw),"min_risk":min_risk,"n_show":n_show}
                    st.rerun()
                elif raw == []:
                    st.warning(f"No posts found for u/{reddit_user.strip()} in the last 6 months.")
    with rB:
        res = st.session_state.reddit_results
        if res is None:
            st.markdown('<div style="text-align:center;padding:4rem 1rem;color:rgba(255,255,255,0.38)"><p>Enter a username and click Analyse Reddit User.</p></div>', unsafe_allow_html=True)
        else:
            df = res["df"]
            st.markdown(f'<h3>u/{res["username"]}</h3>', unsafe_allow_html=True)
            overall_banner(res["overall"], res["n_posts"], res["n_high"], "6 months")
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            s1,s2,s3 = st.tabs(["Timeline","Posts","Socio-Economic"])
            with s1: st.plotly_chart(timeline_chart(df), use_container_width=True)
            with s2:
                filtered = df[df["risk_score"] >= res["min_risk"]]
                render_post_cards(filtered, sub_col="subreddit", url_col="url", type_col="type", n=res["n_show"])
            with s3: render_socio(res["signals"])

with tab_video:
    vA, vB = st.columns([1, 1.4])
    with vA:
        st.markdown('<h2>Video Analysis</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.6)">Paste any public video URL. Supports TikTok, Facebook, Instagram, Twitter/X, YouTube, Vimeo, Twitch, and 1000+ other sites.</p>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        video_url = st.text_input("Video URL", placeholder="https://www.tiktok.com/@user/video/...", key="video_url_input")
        st.markdown('<p style="font-size:0.7rem;color:rgba(255,255,255,0.45)">Public videos only. First run downloads Whisper tiny model (~75MB).</p>', unsafe_allow_html=True)
        vid_btn = st.button("Transcribe and Analyse", use_container_width=True, key="video_analyse")
        if vid_btn:
            url = video_url.strip()
            if not url:
                st.warning("Please paste a video URL.")
            else:
                st.session_state.video_result = None
                progress = st.progress(0); status = st.empty()
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        status.markdown('<p style="font-size:0.76rem;color:#5eead4">Downloading audio...</p>', unsafe_allow_html=True)
                        progress.progress(15)
                        audio_path = download_audio(url, tmpdir)
                        status.markdown('<p style="font-size:0.76rem;color:#5eead4">Transcribing speech...</p>', unsafe_allow_html=True)
                        progress.progress(50)
                        transcript = transcribe_audio(audio_path)
                    progress.progress(75)
                    if not transcript.strip():
                        st.session_state.video_result = {"ok":False,"reason":"no_speech","url":url}
                    else:
                        status.markdown('<p style="font-size:0.76rem;color:#5eead4">Running Bi-LSTM...</p>', unsafe_allow_html=True)
                        prob, ms = predict_one(clean_text(transcript)); risk = 1.0 - prob
                        progress.progress(100); status.empty()
                        st.session_state.video_result = {"ok":True,"url":url,"transcript":transcript,"prob":prob,"risk":risk,"ms":ms}
                        update_analytics(prob, transcript)
                except RuntimeError as e:
                    progress.progress(100); status.empty()
                    st.session_state.video_result = {"ok":False,"reason":"error","msg":str(e),"url":url}
                st.rerun()
    with vB:
        vr = st.session_state.video_result
        if vr is None:
            st.markdown('<div style="text-align:center;padding:4rem 1rem;color:rgba(255,255,255,0.38)"><p>Paste a video URL and click Transcribe and Analyse.</p><p style="font-size:0.72rem;margin-top:0.4rem">Supported: TikTok · Facebook · Instagram · Twitter/X · YouTube · Vimeo · Twitch · and more</p></div>', unsafe_allow_html=True)
        elif not vr.get("ok"):
            if vr.get("reason") == "no_speech":
                st.warning("No speech detected. The video may be music-only or silent.")
            else:
                st.error(f"{vr.get('msg','Download or transcription failed.')}")
                st.markdown('<p style="font-size:0.72rem;color:rgba(255,255,255,0.45)">Common causes: private video, region-locked, or URL expired.</p>', unsafe_allow_html=True)
        else:
            risk = vr["risk"]; prob = vr["prob"]; lbl, col, _ = risk_label(risk)
            st.markdown('<p class="section-label">Transcript</p>', unsafe_allow_html=True)
            st.markdown(f'<div style="background:rgba(0,0,0,0.22);border-radius:10px;padding:0.55rem 0.75rem;border:1px solid rgba(255,255,255,0.1);font-size:0.76rem;line-height:1.6;color:rgba(255,255,255,0.85);max-height:150px;overflow-y:auto;">{vr["transcript"]}</div>', unsafe_allow_html=True)
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">Prediction</p>', unsafe_allow_html=True)
            r1,r2,r3 = st.columns(3)
            r1.metric("Risk Score", f"{risk:.1%}"); r2.metric("Risk Level", lbl); r3.metric("Latency", f"{vr['ms']:.0f}ms")
            st.plotly_chart(gauge(prob), use_container_width=True)
            if risk >= 0.55:   st.error("CRISIS ALERT — High-risk content detected.")
            elif risk >= 0.35: st.warning("Moderate risk detected.")
            else:              st.success("Low risk detected.")
            dl = f"Video URL: {vr['url']}\n\nTranscript:\n{vr['transcript']}\n\nRisk Score: {risk:.1%}\nRisk Level: {lbl}\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            st.download_button("Download report", dl, file_name="video_report.txt", use_container_width=True)
    vr2 = st.session_state.video_result
    if vr2 and vr2.get("ok"):
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Socio-Economic Signals in Transcript</p>', unsafe_allow_html=True)
        render_socio(detect_socioeconomic([{"text": vr2["transcript"]}]))


with tab_bluesky:
    st.markdown('<h2>Bluesky Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.6)">Fetches 3 months of posts for any public Bluesky account. Requires your Bluesky credentials to authenticate.</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    bA, bB = st.columns([1, 2])
    with bA:
        st.text_input("Bluesky handle to analyse", placeholder="e.g. bsky.app", key="bsky_handle_input")
        st.markdown('<p class="section-label">Your credentials</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.7rem;color:rgba(255,255,255,0.45)">Bluesky Settings &rarr; Privacy &rarr; App Passwords &rarr; Add App Password</p>', unsafe_allow_html=True)
        st.text_input("Your Bluesky handle", placeholder="your.handle.bsky.social", key="bsky_identifier_input")
        st.text_input("App Password", type="password", placeholder="xxxx-xxxx-xxxx-xxxx", key="bsky_password_input")
        st.slider("Min risk score to display", 0.0, 1.0, 0.0, 0.05, key="bsky_min_input")
        st.slider("Max posts to display", 5, 50, 20, 5, key="bsky_n_input")
        if st.button("Analyse Bluesky User", use_container_width=True, key="bsky_go"):
            st.session_state["bsky_pending"] = {"handle":st.session_state.get("bsky_handle_input","").strip(),"identifier":st.session_state.get("bsky_identifier_input","").strip(),"password":st.session_state.get("bsky_password_input","").strip(),"min_risk":st.session_state.get("bsky_min_input",0.0),"n_show":st.session_state.get("bsky_n_input",20)}
    pending = st.session_state.get("bsky_pending")
    if pending:
        st.session_state["bsky_pending"] = None
        handle = pending["handle"]; identifier = pending["identifier"]; password = pending["password"]
        if not handle: st.warning("Enter the Bluesky handle you want to analyse.")
        elif not identifier or not password: st.warning("Enter your Bluesky handle and App Password.")
        else:
            access_token = None; error_msg = None
            with st.spinner("Logging in to Bluesky..."):
                try: access_token = bluesky_login(identifier, password)
                except RuntimeError as e: error_msg = str(e)
            if error_msg:
                st.error(f"Login failed: {error_msg}")
            else:
                raw = []
                with st.spinner(f"Fetching posts for {handle}..."):
                    try: raw = fetch_bluesky(handle, access_token=access_token)
                    except RuntimeError as e: error_msg = str(e)
                    except Exception as e: error_msg = f"Unexpected error: {e}"
                if error_msg: st.error(f"Could not fetch posts: {error_msg}")
                elif not raw: st.warning(f"No posts found for '{handle}' in the last 3 months.")
                else:
                    with st.spinner(f"Running Bi-LSTM on {len(raw)} posts..."):
                        scores = predict_batch([clean_text(p["text"]) for p in raw])
                    df = pd.DataFrame(raw); df["risk_score"] = scores; df["date"] = pd.to_datetime(df["date"], utc=True)
                    st.session_state.bluesky_results = {"handle":handle,"df":df,"overall":float(np.percentile(scores,85)),"n_high":int((scores>=0.55).sum()),"signals":detect_socioeconomic(raw),"n_posts":len(raw),"min_risk":pending["min_risk"],"n_show":pending["n_show"]}
                    st.rerun()
    res = st.session_state.bluesky_results
    if res is None:
        with bB:
            st.markdown('<div style="text-align:center;padding:4rem 1rem;color:rgba(255,255,255,0.38)"><p>Enter a handle and credentials, then click Analyse Bluesky User.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        df = res["df"]; st.markdown(f'<h3>Results for {res["handle"]}</h3>', unsafe_allow_html=True)
        overall_banner(res["overall"], res["n_posts"], res["n_high"], "3 months")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        s1, s2, s3 = st.tabs(["Timeline","Posts","Socio-Economic"])
        with s1: st.plotly_chart(timeline_chart(df), use_container_width=True)
        with s2:
            filtered = df[df["risk_score"] >= res["min_risk"]]
            render_post_cards(filtered, url_col="url", n=res["n_show"])
        with s3: render_socio(res["signals"])

with tab_mastodon:
    mA, mB = st.columns([1, 2])
    with mA:
        st.markdown('<h2>Mastodon Analysis</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.6)">Fetches 3 months of posts for any public Mastodon account. No API key needed.</p>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        mast_handle = st.text_input("Mastodon handle", placeholder="e.g. username@mastodon.social", key="mast_handle")
        mast_min = st.slider("Show posts above risk score", 0.0, 1.0, 0.0, 0.05, key="mast_min")
        mast_n   = st.slider("Max posts to display", 5, 50, 20, 5, key="mast_n")
        mast_btn = st.button("Analyse Mastodon User", use_container_width=True, key="mast_fetch")
        if mast_btn:
            handle = mast_handle.strip()
            if not handle: st.warning("Enter a Mastodon handle (format: username@instance.social).")
            else:
                with st.spinner(f"Fetching posts for {handle}..."):
                    try: raw = fetch_mastodon(handle)
                    except RuntimeError as e: st.error(str(e)); raw = []
                if raw:
                    with st.spinner(f"Running Bi-LSTM on {len(raw)} posts..."):
                        scores = predict_batch([clean_text(p["text"]) for p in raw])
                    df = pd.DataFrame(raw); df["risk_score"] = scores; df["date"] = pd.to_datetime(df["date"], utc=True)
                    st.session_state.mastodon_results = {"handle":handle,"df":df,"overall":float(np.percentile(scores,85)),"n_high":int((scores>=0.55).sum()),"signals":detect_socioeconomic(raw),"n_posts":len(raw),"min_risk":mast_min,"n_show":mast_n}
                    st.rerun()
                elif raw == []: st.warning("No posts found or account is private/not found.")
    with mB:
        res = st.session_state.mastodon_results
        if res is None:
            st.markdown('<div style="text-align:center;padding:4rem 1rem;color:rgba(255,255,255,0.38)"><p>Enter a handle and click Analyse Mastodon User.</p></div>', unsafe_allow_html=True)
        else:
            df = res["df"]; st.markdown(f'<h3>{res["handle"]}</h3>', unsafe_allow_html=True)
            overall_banner(res["overall"], res["n_posts"], res["n_high"], "3 months")
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            s1,s2,s3 = st.tabs(["Timeline","Posts","Socio-Economic"])
            with s1: st.plotly_chart(timeline_chart(df), use_container_width=True)
            with s2:
                filtered = df[df["risk_score"] >= res["min_risk"]]
                render_post_cards(filtered, url_col="url", n=res["n_show"])
            with s3: render_socio(res["signals"])

with tab_youtube:
    yA, yB = st.columns([1, 2])
    with yA:
        st.markdown('<h2>YouTube Channel Analysis</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.6)">Analyses video titles, descriptions, and top comments from a YouTube channel. Requires a free YouTube Data API v3 key.</p>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        yt_channel = st.text_input("YouTube channel URL or @handle", placeholder="https://youtube.com/@channelname", key="yt_channel")
        with st.expander("YouTube API key", expanded=True):
            st.markdown('<p style="font-size:0.7rem">Free at <a href="https://console.cloud.google.com" target="_blank">console.cloud.google.com</a> — enable YouTube Data API v3.</p>', unsafe_allow_html=True)
            yt_key = st.text_input("API Key", value=os.getenv("YOUTUBE_API_KEY",""), type="password", key="yt_key")
        yt_min = st.slider("Show posts above risk score", 0.0, 1.0, 0.0, 0.05, key="yt_min")
        yt_n   = st.slider("Max items to display", 5, 50, 20, 5, key="yt_n")
        yt_btn = st.button("Analyse YouTube Channel", use_container_width=True, key="yt_fetch")
        if yt_btn:
            if not yt_channel.strip(): st.warning("Enter a channel URL or handle.")
            elif not yt_key: st.error("Enter your YouTube API key.")
            else:
                with st.spinner("Fetching YouTube data..."):
                    try: raw = fetch_youtube(yt_channel.strip(), yt_key)
                    except RuntimeError as e: st.error(str(e)); raw = []
                if raw:
                    with st.spinner(f"Running Bi-LSTM on {len(raw)} items..."):
                        scores = predict_batch([clean_text(p["text"]) for p in raw])
                    df = pd.DataFrame(raw); df["risk_score"] = scores; df["date"] = pd.to_datetime(df["date"], utc=True)
                    st.session_state.youtube_results = {"channel":yt_channel.strip(),"df":df,"overall":float(np.percentile(scores,85)),"n_high":int((scores>=0.55).sum()),"signals":detect_socioeconomic(raw),"n_posts":len(raw),"min_risk":yt_min,"n_show":yt_n}
                    st.rerun()
                elif raw == []: st.warning("No content found in the last 3 months.")
    with yB:
        res = st.session_state.youtube_results
        if res is None:
            st.markdown('<div style="text-align:center;padding:4rem 1rem;color:rgba(255,255,255,0.38)"><p>Enter a channel and click Analyse YouTube Channel.</p></div>', unsafe_allow_html=True)
        else:
            df = res["df"]; st.markdown(f'<h3>{res["channel"]}</h3>', unsafe_allow_html=True)
            overall_banner(res["overall"], res["n_posts"], res["n_high"], "3 months")
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            s1,s2,s3 = st.tabs(["Timeline","Posts","Socio-Economic"])
            with s1: st.plotly_chart(timeline_chart(df), use_container_width=True)
            with s2:
                filtered = df[df["risk_score"] >= res["min_risk"]]
                render_post_cards(filtered, url_col="url", n=res["n_show"])
            with s3: render_socio(res["signals"])


with tab_file:
    fA, fB = st.columns([1, 2])
    with fA:
        st.markdown('<h2>File Upload Analysis</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.6)">Upload exported chat logs, journal entries, or any text file.</p>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.72rem;color:rgba(255,255,255,0.55)">Accepted formats</p>', unsafe_allow_html=True)
        st.markdown('<ul style="font-size:0.72rem;color:rgba(255,255,255,0.7);padding-left:1.2rem"><li>.txt — WhatsApp export, journal, any plain text</li><li>.csv — exported tweets, posts, or any table with a text column</li><li>.json — Facebook data archive (posts_1.json) or Twitter/X archive (tweet.js)</li></ul>', unsafe_allow_html=True)
        st.markdown('<div style="background:rgba(13,148,136,0.1);border-radius:8px;padding:0.4rem 0.65rem;border:1px solid rgba(13,148,136,0.25);margin-bottom:0.4rem"><p style="font-size:0.72rem;color:rgba(255,255,255,0.7);margin:0"><strong style="color:#5eead4">Facebook:</strong> Settings &rarr; Your Facebook Information &rarr; Download Your Information &rarr; Posts &rarr; JSON format<br><strong style="color:#5eead4">Twitter/X:</strong> Settings &rarr; Your Account &rarr; Download an archive &rarr; tweet.js</p></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload file", type=["txt","csv","json"], label_visibility="collapsed")
        file_min = st.slider("Show entries above risk score", 0.0, 1.0, 0.0, 0.05, key="file_min")
        file_n   = st.slider("Max entries to display", 5, 100, 30, 5, key="file_n")
        file_btn = st.button("Analyse File", use_container_width=True, key="file_analyse")
        if file_btn:
            if not uploaded: st.warning("Please upload a file first.")
            else:
                with st.spinner("Parsing file..."):
                    try: raw = parse_uploaded_file(uploaded)
                    except Exception as e: st.error(f"Could not parse file: {e}"); raw = []
                if raw:
                    with st.spinner(f"Running Bi-LSTM on {len(raw)} entries..."):
                        scores = predict_batch([clean_text(p["text"]) for p in raw])
                    df = pd.DataFrame(raw); df["risk_score"] = scores; df["date"] = pd.to_datetime(df["date"], utc=True)
                    st.session_state.file_results = {"filename":uploaded.name,"df":df,"overall":float(np.percentile(scores,85)),"n_high":int((scores>=0.55).sum()),"signals":detect_socioeconomic(raw),"n_posts":len(raw),"min_risk":file_min,"n_show":file_n}
                    st.rerun()
                elif raw == []: st.warning("No readable text found in the file.")
    with fB:
        res = st.session_state.file_results
        if res is None:
            st.markdown('<div style="text-align:center;padding:4rem 1rem;color:rgba(255,255,255,0.38)"><p>Upload a file and click Analyse File.</p></div>', unsafe_allow_html=True)
        else:
            df = res["df"]; st.markdown(f'<h3>{res["filename"]}</h3>', unsafe_allow_html=True)
            overall_banner(res["overall"], res["n_posts"], res["n_high"], "file")
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            s1,s2,s3 = st.tabs(["Timeline","Entries","Socio-Economic"])
            with s1: st.plotly_chart(timeline_chart(df), use_container_width=True)
            with s2:
                filtered = df[df["risk_score"] >= res["min_risk"]]
                render_post_cards(filtered, n=res["n_show"])
            with s3: render_socio(res["signals"])

with tab_facebook:
    st.markdown('<h2>Facebook Public Profile Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.6)">Scrapes public posts from a Facebook profile using a headless browser. Only works for profiles with public post visibility.</p>', unsafe_allow_html=True)
    st.markdown('<div style="background:rgba(245,158,11,0.1);border-radius:8px;padding:0.45rem 0.7rem;border:1px solid rgba(245,158,11,0.3);margin-bottom:0.5rem"><p style="color:#fbbf24;font-size:0.74rem;margin:0">Only publicly visible posts are accessed. Research use under ethics approval TUM-SERC MSC/028/2025A.</p></div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    fA, fB = st.columns([1, 2])
    with fA:
        st.text_input("Facebook profile URL", placeholder="https://www.facebook.com/username", key="fb_url_input")
        st.slider("Months to analyse", 1, 6, 3, 1, key="fb_months")
        st.slider("Show posts above risk score", 0.0, 1.0, 0.0, 0.05, key="fb_min")
        st.slider("Max posts to display", 5, 50, 20, 5, key="fb_n")
        if st.button("Scrape and Analyse", use_container_width=True, key="fb_go"):
            st.session_state["fb_triggered"] = True
    if st.session_state.get("fb_triggered"):
        fb_url_val = st.session_state.get("fb_url_input","").strip()
        fb_mon_val = st.session_state.get("fb_months", 3)
        fb_min_val = st.session_state.get("fb_min", 0.0)
        fb_n_val   = st.session_state.get("fb_n", 20)
        st.session_state["fb_triggered"] = False
        if not fb_url_val: st.warning("Enter a Facebook profile URL.")
        elif "facebook.com" not in fb_url_val: st.warning("Enter a full Facebook URL e.g. https://www.facebook.com/username")
        else:
            raw = []; err_msg = None
            st.info(f"Headless browser starting — scraping {fb_url_val} ...")
            with st.spinner("This takes 30-60 seconds. Please wait..."):
                try: raw = scrape_facebook_public(fb_url_val, months=fb_mon_val)
                except RuntimeError as e: err_msg = str(e)
                except Exception as e: err_msg = f"Unexpected error: {e}"
            if err_msg:
                st.error(f"Scraping failed: {err_msg}")
                st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.5)">Common causes: profile is private, Facebook blocked the request, or the URL is incorrect.</p>', unsafe_allow_html=True)
            elif not raw: st.warning("No public posts found. The profile may be private or Facebook blocked the request.")
            else:
                with st.spinner(f"Running Bi-LSTM on {len(raw)} posts..."):
                    scores = predict_batch([clean_text(p["text"]) for p in raw])
                df = pd.DataFrame(raw); df["risk_score"] = scores; df["date"] = pd.to_datetime(df["date"], utc=True)
                st.session_state.facebook_results = {"url":fb_url_val,"df":df,"overall":float(np.percentile(scores,85)),"n_high":int((scores>=0.55).sum()),"signals":detect_socioeconomic(raw),"n_posts":len(raw),"min_risk":fb_min_val,"n_show":fb_n_val}
                st.rerun()
    res = st.session_state.facebook_results
    if res is None:
        with fB:
            st.markdown('<div style="text-align:center;padding:4rem 1rem;color:rgba(255,255,255,0.38)"><p>Enter a public Facebook profile URL and click Scrape and Analyse.</p><p style="font-size:0.72rem;margin-top:0.5rem">Only public posts are accessible.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f'<h3>Results for {res["url"]}</h3>', unsafe_allow_html=True)
        overall_banner(res["overall"], res["n_posts"], res["n_high"], "months")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        s1, s2 = st.tabs(["Posts","Socio-Economic"])
        with s1:
            filtered = res["df"][res["df"]["risk_score"] >= res["min_risk"]]
            render_post_cards(filtered, url_col="url", n=res["n_show"])
        with s2: render_socio(res["signals"])

with tab_twitter:
    st.markdown('<h2>Twitter / X Public Profile Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.6)">Scrapes public tweets from a Twitter/X profile using a headless browser. Only works for public profiles.</p>', unsafe_allow_html=True)
    st.markdown('<div style="background:rgba(245,158,11,0.1);border-radius:8px;padding:0.45rem 0.7rem;border:1px solid rgba(245,158,11,0.3);margin-bottom:0.5rem"><p style="color:#fbbf24;font-size:0.74rem;margin:0">Twitter/X increasingly requires login to view profiles. If scraping fails, use the File Upload tab with a Twitter data archive instead.</p></div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    tA, tB = st.columns([1, 2])
    with tA:
        st.text_input("Twitter/X profile URL", placeholder="https://x.com/username", key="tw_url_input")
        st.slider("Show posts above risk score", 0.0, 1.0, 0.0, 0.05, key="tw_min")
        st.slider("Max posts to display", 5, 50, 20, 5, key="tw_n")
        if st.button("Scrape and Analyse", use_container_width=True, key="tw_go"):
            st.session_state["tw_triggered"] = True
    if st.session_state.get("tw_triggered"):
        tw_url_val = st.session_state.get("tw_url_input","").strip()
        tw_min_val = st.session_state.get("tw_min", 0.0)
        tw_n_val   = st.session_state.get("tw_n", 20)
        st.session_state["tw_triggered"] = False
        if not tw_url_val: st.warning("Enter a Twitter/X profile URL.")
        elif "twitter.com" not in tw_url_val and "x.com" not in tw_url_val: st.warning("Enter a valid URL e.g. https://x.com/username")
        else:
            raw = []; err_msg = None
            st.info(f"Starting browser scrape of {tw_url_val}...")
            with st.spinner(f"Opening headless browser and scraping {tw_url_val}..."):
                try: raw = scrape_twitter_public(tw_url_val)
                except RuntimeError as e: err_msg = str(e)
                except Exception as e: err_msg = f"Unexpected error: {e}"
            if err_msg:
                st.error(f"Scraping failed: {err_msg}")
                st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.5)">Twitter/X may require login. Use File Upload with a Twitter archive instead.</p>', unsafe_allow_html=True)
            elif not raw: st.warning("No tweets found. Profile may be private. Try File Upload with a Twitter archive.")
            else:
                with st.spinner(f"Running Bi-LSTM on {len(raw)} tweets..."):
                    scores = predict_batch([clean_text(p["text"]) for p in raw])
                df = pd.DataFrame(raw); df["risk_score"] = scores; df["date"] = pd.to_datetime(df["date"], utc=True)
                st.session_state.twitter_results = {"url":tw_url_val,"df":df,"overall":float(np.percentile(scores,85)),"n_high":int((scores>=0.55).sum()),"signals":detect_socioeconomic(raw),"n_posts":len(raw),"min_risk":tw_min_val,"n_show":tw_n_val}
                st.rerun()
    res = st.session_state.twitter_results
    if res is None:
        with tB:
            st.markdown('<div style="text-align:center;padding:4rem 1rem;color:rgba(255,255,255,0.38)"><p>Enter a public Twitter/X URL and click Scrape and Analyse.</p><p style="font-size:0.72rem;margin-top:0.5rem">If login is required, use File Upload with a Twitter archive.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f'<h3>Results for {res["url"]}</h3>', unsafe_allow_html=True)
        overall_banner(res["overall"], res["n_posts"], res["n_high"], "3 months")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        s1, s2 = st.tabs(["Posts","Socio-Economic"])
        with s1:
            filtered = res["df"][res["df"]["risk_score"] >= res["min_risk"]]
            render_post_cards(filtered, url_col="url", n=res["n_show"])
        with s2: render_socio(res["signals"])


with tab_unified:
    st.markdown('<h2>Multi-Platform Unified Risk Profile</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.6)">Combines results from all platforms you have already analysed in this session into one unified risk profile.</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    platform_results = {}
    if st.session_state.reddit_results:   platform_results["Reddit"]     = st.session_state.reddit_results
    if st.session_state.bluesky_results:  platform_results["Bluesky"]    = st.session_state.bluesky_results
    if st.session_state.mastodon_results: platform_results["Mastodon"]   = st.session_state.mastodon_results
    if st.session_state.youtube_results:  platform_results["YouTube"]    = st.session_state.youtube_results
    if st.session_state.file_results:     platform_results["File Upload"] = st.session_state.file_results
    if st.session_state.facebook_results: platform_results["Facebook"]   = st.session_state.facebook_results
    if st.session_state.twitter_results:  platform_results["Twitter/X"]  = st.session_state.twitter_results
    if st.session_state.video_result and st.session_state.video_result.get("ok"):
        vr = st.session_state.video_result
        platform_results["Video"] = {"overall":vr["risk"],"n_posts":1,"n_high":1 if vr["risk"]>=0.55 else 0}
    if not platform_results:
        st.markdown('<div style="text-align:center;padding:3rem 1rem;color:rgba(255,255,255,0.4)"><p>No platforms analysed yet. Go to each tab and run an analysis first.</p></div>', unsafe_allow_html=True)
    else:
        rows = []; all_scores = []
        for platform, res in platform_results.items():
            overall = res["overall"]; lbl, col, _ = risk_label(overall)
            rows.append({"Platform":platform,"Posts":res.get("n_posts",1),"Overall Risk":f"{overall:.1%}","High-Risk Posts":res.get("n_high",0),"Risk Level":lbl})
            all_scores.append(overall)
        unified_score = float(np.mean(all_scores)); unified_lbl, unified_col, _ = risk_label(unified_score)
        u1,u2,u3 = st.columns(3)
        u1.metric("Unified Risk Score", f"{unified_score:.1%}")
        u2.metric("Platforms Analysed", str(len(platform_results)))
        u3.metric("Unified Risk Level", unified_lbl)
        st.markdown(f'<div style="display:inline-block;padding:5px 16px;border-radius:999px;background:{unified_col}22;color:{unified_col};border:1.5px solid {unified_col};font-weight:700;font-size:0.88rem;margin:4px 0">{unified_lbl} — Unified across {len(platform_results)} platform(s)</div>', unsafe_allow_html=True)
        if unified_score >= 0.55:
            st.error("CRISIS ALERT — Elevated risk detected across multiple platforms.")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Platform Breakdown</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_bar(x=[r["Platform"] for r in rows],y=[float(r["Overall Risk"].rstrip("%"))/100 for r in rows],marker_color=["#22c55e" if float(r["Overall Risk"].rstrip("%"))/100<0.35 else "#f59e0b" if float(r["Overall Risk"].rstrip("%"))/100<0.55 else "#f97316" if float(r["Overall Risk"].rstrip("%"))/100<0.75 else "#ef4444" for r in rows],text=[r["Overall Risk"] for r in rows],textposition="outside",textfont_color="rgba(255,255,255,0.8)")
        fig.add_hline(y=unified_score,line_dash="dot",line_color="white",annotation_text=f"Unified avg: {unified_score:.1%}",annotation_font_color="white",annotation_font_size=10)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(19,34,51,0.5)",font_color="rgba(255,255,255,0.7)",yaxis=dict(tickformat=".0%",range=[0,1.1],gridcolor="rgba(255,255,255,0.07)",color="rgba(255,255,255,0.6)"),xaxis=dict(color="rgba(255,255,255,0.7)"),margin=dict(l=20,r=20,t=30,b=20),height=280,showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="section-label">Detail Table</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows).set_index("Platform"), use_container_width=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Combined Socio-Economic Signals</p>', unsafe_allow_html=True)
        combined_signals = {cat: [] for cat in SOCIOECONOMIC_KEYWORDS}
        for res in platform_results.values():
            if "signals" in res:
                for cat, kws in res["signals"].items():
                    combined_signals[cat] = list(set(combined_signals[cat] + kws))
        render_socio(combined_signals)
        report_lines = [f"MindGuard Unified Risk Report\n{'='*40}\n",f"Unified Risk Score: {unified_score:.1%}  ({unified_lbl})\n",f"Platforms analysed: {', '.join(platform_results.keys())}\n\n"]
        for r in rows:
            report_lines.append(f"{r['Platform']}: {r['Overall Risk']} — {r['Risk Level']} ({r['Posts']} posts, {r['High-Risk Posts']} high-risk)\n")
        report_lines.append(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        st.download_button("Download unified report", "".join(report_lines), file_name="mindguard_unified_report.txt", use_container_width=True)

with tab_resources:
    st.markdown('<h2>Crisis Resources</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.6)">Select your country or US state to see local crisis resources.</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    rc1, rc2 = st.columns([1, 1])
    with rc1:
        country_options = list(RESOURCES.keys()) + ["USA — Select a State"]
        selected_country = st.selectbox("Country / Region", country_options, key="res_region")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        if selected_country == "USA — Select a State":
            selected_state = st.selectbox("Select your state", sorted(US_STATE_RESOURCES.keys()), key="res_state")
            st.markdown(f'<p class="section-label">Resources for {selected_state}</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.72rem;font-weight:600;color:rgba(255,255,255,0.55);margin:0.3rem 0 0.15rem">National (available in all states)</p>', unsafe_allow_html=True)
            for r in RESOURCES["USA (National)"]:
                render_resource_card(r, border_color="#0d9488")
            st.markdown(f'<p style="font-size:0.72rem;font-weight:600;color:rgba(255,255,255,0.55);margin:0.5rem 0 0.15rem">State-specific — {selected_state}</p>', unsafe_allow_html=True)
            for r in US_STATE_RESOURCES.get(selected_state, []):
                render_resource_card(r, border_color="#7c3aed")
        else:
            render_resources(selected_country)
        st.markdown('<div style="margin-top:0.7rem;padding:0.5rem 0.65rem;background:rgba(239,68,68,0.08);border-radius:8px;border:1px solid rgba(239,68,68,0.25)"><p style="color:#fca5a5;font-size:0.76rem;margin:0">If someone is in immediate danger, call emergency services immediately. MindGuard is a research tool — it does not replace clinical assessment.</p></div>', unsafe_allow_html=True)
    with rc2:
        st.markdown('<h2>About MindGuard</h2>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.76rem;line-height:1.85;color:rgba(255,255,255,0.78)"><p class="section-label">Model Architecture</p><p>Mental-RoBERTa (mental/mental-roberta-base) — domain-specific transformer pre-trained on mental health text. Fine-tuned on 12,656 annotated Reddit posts. Accuracy: 92.5% | ROC-AUC: 0.9813</p><p class="section-label">Risk Tiers</p><p>Low &lt;35% &nbsp;&middot;&nbsp; Moderate 35-55% &nbsp;&middot;&nbsp; High 55-75% &nbsp;&middot;&nbsp; Critical &gt;75%</p><p class="section-label">Ethics</p><p>Ethics approval: TUM-SERC MSC/028/2025A<br>NACOSTI Application #535883<br>Consent-first. No data stored between sessions. For research use only.</p></div>', unsafe_allow_html=True)