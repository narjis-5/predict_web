# -*- coding: utf-8 -*-
"""
LOGI-PREDICT — Aide a la decision logistique
============================================
Application Streamlit multilingue (FR / EN / AR + RTL) d'AIDE A LA DECISION.

Architecture
------------
1. CHARGEMENT (aucun reentrainement) : on charge le bundle de regression deja
   choisi (delivery_duration_model_bundle.joblib : preprocessing anti-fuite +
   HistGradientBoostingRegressor qui predit la duree REELLE de transit). Si un
   bundle de classification de la duree existe aussi
   (delivery_duration_class_model_bundle.joblib), on le charge pour afficher la
   CLASSE de duree (standard / longue) et sa probabilite ; sinon on l'ignore
   sans erreur.
2. PIPELINE (identique au notebook) : conversion des dates -> calcul de
   estimated_transit_days -> mois et jour de semaine d'expedition -> suppression
   des lignes incoherentes (durees negatives / dates manquantes) -> construction
   de la matrice d'entree a partir du SCHEMA du bundle -> application du pipeline.
   La colonne actual_delivery_date est OPTIONNELLE et ne sert qu'a comparer le
   predit au reel : elle n'entre JAMAIS dans le modele (anti-fuite).
3. DECISION : predicted_duration_days -> eta_gap_days (predit - ETA) ->
   risk_score (indicateur de PRIORISATION 0-100 %, PAS une probabilite calibree)
   -> tampon ETA recommande par axe, niveau de confiance selon le volume,
   action recommandee.

L'application NE reentraine RIEN et NE teste AUCUN modele : elle applique le
bundle deja entraine. Aucune metrique n'est inventee : tout provient du bundle,
de model_report.json et des donnees chargees.

Lancement :  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Constantes et chemins
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
BUNDLE_PATH = HERE / "delivery_duration_model_bundle.joblib"
CLASS_BUNDLE_PATH = HERE / "delivery_duration_class_model_bundle.joblib"
REPORT_PATH = HERE / "model_report.json"
DATA_PATH = HERE / "nigerian_retail_and_ecommerce_supply_chain_logistics_data.parquet"

# Colonnes brutes minimales requises pour produire une prediction
REQUIRED_RAW = [
    "ship_date", "expected_delivery_date", "origin_city", "destination_city",
    "logistics_company", "supplier_name", "quantity", "shipping_cost_ngn",
]

PALETTE = {
    "ink": "#0f1b2d",
    "slate": "#1f3a5f",
    "accent": "#e8743b",      # accent chaud
    "accent_soft": "#f4a261",
    "teal": "#1f7a8c",
    "low": "#1f9d6b",
    "medium": "#d8902a",
    "high": "#d24b54",
    "muted": "#64748b",
}

st.set_page_config(
    page_title="LOGI-PREDICT | Aide a la decision logistique",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dictionnaire de traductions : aucune chaine d'interface hors de ce dict
# ---------------------------------------------------------------------------
TEXT = {
    "fr": {
        "language": "Langue",
        "brand": "LOGI-PREDICT",
        "tagline": "Aide a la decision logistique",
        "hero_title": "Pilotage predictif des livraisons e-commerce",
        "hero_body": "Importez vos expeditions : l'outil predit la duree reelle probable, la compare a l'ETA promise, et la transforme en actions operationnelles par axe et par transporteur.",
        "honest_phrase": "L'application estime la duree reelle probable de chaque livraison et sa classe de duree, recalibre l'ETA promise et hierarchise les axes et transporteurs a surveiller pour fiabiliser les promesses de livraison. Le score de risque est un indicateur de priorisation operationnelle.",
        # Sidebar
        "service_level": "Niveau de service vise",
        "service_balanced": "Equilibre (80 %)",
        "service_reliable": "Fiable (90 %)",
        "service_premium": "Premium (95 %)",
        "upload": "Deposer un fichier",
        "upload_help": "Formats acceptes : CSV, TXT/TSV, Excel (.xlsx), Parquet, JSON.",
        # Accueil
        "home_what_title": "Ce que fait l'outil",
        "home_what_body": "Il charge un modele deja entraine et l'applique a vos expeditions pour estimer la duree reelle de livraison, l'ecart a l'ETA, un score de priorisation et un tampon ETA recommande par axe et transporteur. Il ne reentraine rien.",
        "home_format_title": "Format & colonnes attendues",
        "home_format_body": "Colonnes obligatoires :",
        "home_format_optional": "Optionnelle (comparaison reel vs predit, jamais en entree du modele) : actual_delivery_date.",
        "home_format_files": "Fichiers : CSV, TXT/TSV, Excel, Parquet, JSON.",
        "home_limits_title": "Limites honnetes",
        "home_limits_body": "Le score de risque est un indicateur de PRIORISATION (0-100 %), pas une probabilite calibree de retard individuel. La valeur de l'outil est le recalibrage de l'ETA et la priorisation, pas la prediction du retard d'un colis isole.",
        "example_button": "Telecharger un fichier-exemple",
        "demo_button": "Voir une demonstration (donnees de reference)",
        "schema_col": "colonne",
        "schema_required": "obligatoire",
        "schema_role": "role",
        "schema_yes": "oui",
        "schema_optional": "optionnelle",
        "waiting": "Deposez un fichier dans la barre laterale pour demarrer l'analyse, ou telechargez le fichier-exemple ci-dessus.",
        # Erreurs / validation
        "model_missing": "Bundle de modele introuvable (delivery_duration_model_bundle.joblib).",
        "format_error": "Lecture impossible. Utilisez un fichier tabulaire valide : CSV, Excel, Parquet ou JSON.",
        "missing_cols": "Colonnes obligatoires absentes",
        "found_cols": "Colonnes trouvees dans votre fichier",
        "no_usable": "Aucune ligne exploitable apres preparation (verifiez les dates).",
        # Onglets
        "tabs": ["Vue generale", "Aide a la decision", "Risques & axes", "Preparation", "Modele & interpretation", "Assistant"],
        # Cartes / metriques
        "raw_rows": "Lignes importees",
        "usable_rows": "Lignes exploitables",
        "predicted_late": "Part depassant l'ETA",
        "avg_risk": "Score de risque moyen",
        "avg_buffer": "Tampon ETA moyen",
        "high_risk": "Expeditions a risque eleve",
        "overview": "Vue generale des donnees importees",
        "preview": "Apercu des predictions, ligne par ligne",
        "weather_note": "Le modele deploye n'utilise aucune variable meteo (schema du bundle).",
        # Decision
        "decision_title": "Aide a la decision operationnelle",
        "decision_intro": "Le tampon ETA recommande est le nombre de jours a ajouter a l'ETA actuel pour atteindre le niveau de service choisi.",
        "buffer_by_group": "Tampon ETA recommande par axe et transporteur",
        "recommendations": "Recommandations issues des donnees",
        "download_decision": "Telecharger la table de decision (CSV)",
        "download_predictions": "Telecharger les predictions (CSV)",
        # Risques & axes
        "routes_at_risk": "Axes les plus sensibles",
        "carrier_risk": "Risque moyen par transporteur",
        "matrix_title": "Matrice risque x tampon ETA",
        "gap_dist_title": "Distribution de la marge ETA (predit - ETA)",
        "service_slider": "Niveau de service (recalcule le tampon en direct)",
        # Preparation
        "preparation": "Tracabilite de la preparation",
        "prep_initial": "Lignes initiales",
        "prep_removed": "Lignes supprimees (incoherentes)",
        "prep_usable": "Lignes exploitables",
        "prep_actual": "Date reelle disponible",
        "prep_weather": "Variables meteo utilisees par le modele",
        # Modele
        "model_title": "Modele deploye et interpretation",
        "m_mae": "MAE (jours)",
        "m_model": "Modele",
        "m_rows": "Lignes d'apprentissage",
        "m_r2": "R2 (test)",
        "v_title": "Validation experte",
        "v_temporal": "R2 temporel",
        "v_auc": "AUC regle de risque",
        "v_falsealarm": "Taux de fausses alertes",
        "v_baserate": "Taux reel de depassement",
        "importance_title": "Importance des variables (chute de R2 par permutation)",
        "model_honesty": "La validation temporelle confirme la stabilite de la prediction de duree ; l'AUC proche de 0,5 de la regle de risque montre que le modele ne distingue pas les retards individuels — la valeur reside dans le recalibrage de l'ETA et la priorisation.",
        "class_section": "Classe de duree de livraison (modele de classification)",
        "class_present": "Modele de classe de duree charge",
        "class_absent": "Aucun modele de classe de duree (section ignoree).",
        "class_acc": "Accuracy",
        "class_auc": "AUC-ROC",
        "class_f1": "F1 (macro)",
        "class_long_share": "Part en classe longue",
        "scatter3d_title": "Vue 3D : duree predite × ETA × score de risque",
        "scatter3d_note": "Chaque point est une expedition ; faites pivoter la vue pour explorer la relation duree predite / ETA / risque.",
        "class_dist_title": "Repartition des classes de duree",
        "confusion_title": "Matrice de confusion (test scelle)",
        "conf_pred": "predit", "conf_real": "reel",
        "class_no_model": "Modele de classe de duree non charge.",
        # Assistant
        "assistant_title": "Assistant d'interpretation des resultats",
        "assistant_hint": "Posez une question sur les axes, transporteurs, tampons ETA ou le score de priorisation.",
        "assistant_nature": "Assistant local, deterministe et a base de regles : il ne fait appel a aucune IA ni API externe ; il repond uniquement a partir des resultats du fichier charge.",
        "assistant_placeholder": "Exemple : quel axe faut-il surveiller en premier ?",
        # Niveaux
        "low": "Faible", "medium": "Moyen", "high": "Eleve",
        "action_ok": "Maintenir", "action_monitor": "Surveiller",
        "action_buffer": "Ajouter tampon ETA", "action_escalate": "Priorite operationnelle",
        "confidence_high": "Forte", "confidence_medium": "Moyenne", "confidence_low": "Faible",
        "class_standard": "standard", "class_long": "longue",
        "day_suffix": "j",
        # Colonnes affichees
        "col_route": "axe", "col_carrier": "transporteur", "col_volume": "volume",
        "col_riskmean": "risque moyen", "col_overrun": "taux depassement ETA",
        "col_buffer": "tampon ETA (j)", "col_durmean": "duree predite moy. (j)",
        "col_etamean": "ETA moyenne (j)", "col_conf": "confiance", "col_action": "action",
    },
    "en": {
        "language": "Language",
        "brand": "LOGI-PREDICT",
        "tagline": "Logistics decision support",
        "hero_title": "Predictive control for e-commerce deliveries",
        "hero_body": "Upload your shipments: the tool predicts the likely real duration, compares it to the promised ETA, and turns it into operational actions by lane and carrier.",
        "honest_phrase": "The app estimates each shipment's likely real duration and its duration class, recalibrates the promised ETA, and ranks the lanes and carriers to watch in order to make delivery promises more reliable. The risk score is an operational prioritization indicator.",
        "service_level": "Target service level",
        "service_balanced": "Balanced (80%)",
        "service_reliable": "Reliable (90%)",
        "service_premium": "Premium (95%)",
        "upload": "Upload a file",
        "upload_help": "Accepted formats: CSV, TXT/TSV, Excel (.xlsx), Parquet, JSON.",
        "home_what_title": "What the tool does",
        "home_what_body": "It loads an already-trained model and applies it to your shipments to estimate the real delivery duration, the gap to the ETA, a prioritization score and a recommended ETA buffer per lane and carrier. It never retrains.",
        "home_format_title": "Format & expected columns",
        "home_format_body": "Required columns:",
        "home_format_optional": "Optional (actual vs predicted comparison, never a model input): actual_delivery_date.",
        "home_format_files": "Files: CSV, TXT/TSV, Excel, Parquet, JSON.",
        "home_limits_title": "Honest limits",
        "home_limits_body": "The risk score is a PRIORITIZATION indicator (0-100%), not a calibrated probability of individual delay. The tool's value is ETA recalibration and prioritization, not predicting a single parcel's delay.",
        "example_button": "Download an example file",
        "demo_button": "See a demonstration (reference data)",
        "schema_col": "column",
        "schema_required": "required",
        "schema_role": "role",
        "schema_yes": "yes",
        "schema_optional": "optional",
        "waiting": "Upload a file from the sidebar to start, or download the example file above.",
        "model_missing": "Model bundle not found (delivery_duration_model_bundle.joblib).",
        "format_error": "Could not read the file. Use a valid tabular file: CSV, Excel, Parquet or JSON.",
        "missing_cols": "Missing required columns",
        "found_cols": "Columns found in your file",
        "no_usable": "No usable rows after preparation (check the dates).",
        "tabs": ["Overview", "Decision aid", "Risks & lanes", "Preparation", "Model & interpretation", "Assistant"],
        "raw_rows": "Imported rows",
        "usable_rows": "Usable rows",
        "predicted_late": "Share exceeding ETA",
        "avg_risk": "Average risk score",
        "avg_buffer": "Average ETA buffer",
        "high_risk": "High-risk shipments",
        "overview": "Overview of imported data",
        "preview": "Row-by-row prediction preview",
        "weather_note": "The deployed model uses no weather features (bundle schema).",
        "decision_title": "Operational decision support",
        "decision_intro": "The recommended ETA buffer is the number of days to add to the current ETA to reach the selected service level.",
        "buffer_by_group": "Recommended ETA buffer by lane and carrier",
        "recommendations": "Data-driven recommendations",
        "download_decision": "Download decision table (CSV)",
        "download_predictions": "Download predictions (CSV)",
        "routes_at_risk": "Most sensitive lanes",
        "carrier_risk": "Average risk by carrier",
        "matrix_title": "Risk x ETA buffer matrix",
        "gap_dist_title": "ETA gap distribution (predicted - ETA)",
        "service_slider": "Service level (recomputes the buffer live)",
        "preparation": "Preparation traceability",
        "prep_initial": "Initial rows",
        "prep_removed": "Removed rows (inconsistent)",
        "prep_usable": "Usable rows",
        "prep_actual": "Actual date available",
        "prep_weather": "Weather features used by the model",
        "model_title": "Deployed model and interpretation",
        "m_mae": "MAE (days)",
        "m_model": "Model",
        "m_rows": "Training rows",
        "m_r2": "R2 (test)",
        "v_title": "Expert validation",
        "v_temporal": "Temporal R2",
        "v_auc": "Risk-rule AUC",
        "v_falsealarm": "False-alarm rate",
        "v_baserate": "Actual overrun rate",
        "importance_title": "Feature importance (permutation R2 drop)",
        "model_honesty": "Temporal validation confirms the stability of the duration prediction; the risk-rule AUC near 0.5 shows the model does not distinguish individual delays — the value lies in ETA recalibration and prioritization.",
        "class_section": "Delivery duration class (classification model)",
        "class_present": "Duration-class model loaded",
        "class_absent": "No duration-class model (section skipped).",
        "class_acc": "Accuracy",
        "class_auc": "ROC-AUC",
        "class_f1": "F1 (macro)",
        "class_long_share": "Share in long class",
        "scatter3d_title": "3D view: predicted duration × ETA × risk score",
        "scatter3d_note": "Each point is a shipment; rotate the view to explore predicted duration / ETA / risk.",
        "class_dist_title": "Duration class distribution",
        "confusion_title": "Confusion matrix (sealed test)",
        "conf_pred": "predicted", "conf_real": "actual",
        "class_no_model": "Duration-class model not loaded.",
        "assistant_title": "Results interpretation assistant",
        "assistant_hint": "Ask about lanes, carriers, ETA buffers or the prioritization score.",
        "assistant_nature": "Local, deterministic, rule-based assistant: it uses no AI or external API; it answers only from the uploaded file results.",
        "assistant_placeholder": "Example: which lane should we monitor first?",
        "low": "Low", "medium": "Medium", "high": "High",
        "action_ok": "Maintain", "action_monitor": "Monitor",
        "action_buffer": "Add ETA buffer", "action_escalate": "Operational priority",
        "confidence_high": "High", "confidence_medium": "Medium", "confidence_low": "Low",
        "class_standard": "standard", "class_long": "long",
        "day_suffix": "d",
        "col_route": "lane", "col_carrier": "carrier", "col_volume": "volume",
        "col_riskmean": "average risk", "col_overrun": "ETA overrun rate",
        "col_buffer": "ETA buffer (d)", "col_durmean": "avg predicted duration (d)",
        "col_etamean": "avg ETA (d)", "col_conf": "confidence", "col_action": "action",
    },
    "ar": {
        "language": "اللغة",
        "brand": "LOGI-PREDICT",
        "tagline": "دعم القرار اللوجستي",
        "hero_title": "قيادة تنبؤية لتسليمات التجارة الإلكترونية",
        "hero_body": "ارفع شحناتك: تتوقع الأداة المدة الفعلية المرجحة، تقارنها بموعد ETA الموعود، وتحولها إلى إجراءات تشغيلية حسب المسار وشركة النقل.",
        "honest_phrase": "يقدّر التطبيق المدة الفعلية المرجحة لكل شحنة وصنف مدتها، ويعيد ضبط موعد ETA الموعود، ويرتّب المسارات وشركات النقل الواجب مراقبتها لجعل وعود التسليم أكثر موثوقية. درجة الخطر مؤشر لترتيب الأولويات التشغيلية.",
        "service_level": "مستوى الخدمة المطلوب",
        "service_balanced": "متوازن (80%)",
        "service_reliable": "موثوق (90%)",
        "service_premium": "ممتاز (95%)",
        "upload": "رفع ملف",
        "upload_help": "الصيغ المقبولة: CSV و TXT/TSV و Excel و Parquet و JSON.",
        "home_what_title": "ماذا تفعل الأداة",
        "home_what_body": "تحمّل نموذجا مدربا مسبقا وتطبّقه على شحناتك لتقدير المدة الفعلية للتسليم، الفارق عن ETA، درجة أولوية، وهامش ETA مقترح لكل مسار وشركة نقل. لا تعيد التدريب إطلاقا.",
        "home_format_title": "الصيغة والأعمدة المطلوبة",
        "home_format_body": "الأعمدة الإلزامية:",
        "home_format_optional": "اختياري (مقارنة الفعلي بالمتوقع، لا يدخل أبدا في النموذج): actual_delivery_date.",
        "home_format_files": "الملفات: CSV و TXT/TSV و Excel و Parquet و JSON.",
        "home_limits_title": "حدود صريحة",
        "home_limits_body": "درجة الخطر مؤشر لترتيب الأولويات (0-100%)، وليست احتمالا معايرا لتأخير فردي. قيمة الأداة هي إعادة ضبط ETA وترتيب الأولويات، لا التنبؤ بتأخير شحنة بعينها.",
        "example_button": "تحميل ملف نموذجي",
        "demo_button": "عرض توضيحي (بيانات مرجعية)",
        "schema_col": "العمود",
        "schema_required": "إلزامي",
        "schema_role": "الدور",
        "schema_yes": "نعم",
        "schema_optional": "اختياري",
        "waiting": "ارفع ملفا من الشريط الجانبي للبدء، أو حمّل الملف النموذجي أعلاه.",
        "model_missing": "حزمة النموذج غير موجودة (delivery_duration_model_bundle.joblib).",
        "format_error": "تعذرت قراءة الملف. استخدم ملفا جدوليا صالحا: CSV أو Excel أو Parquet أو JSON.",
        "missing_cols": "أعمدة إلزامية ناقصة",
        "found_cols": "الأعمدة الموجودة في ملفك",
        "no_usable": "لا توجد صفوف قابلة للاستعمال بعد التحضير (تحقق من التواريخ).",
        "tabs": ["نظرة عامة", "دعم القرار", "المخاطر والمسارات", "التحضير", "النموذج والتفسير", "المساعد"],
        "raw_rows": "الصفوف المستوردة",
        "usable_rows": "الصفوف القابلة للاستعمال",
        "predicted_late": "نسبة تجاوز ETA",
        "avg_risk": "متوسط درجة الخطر",
        "avg_buffer": "متوسط هامش ETA",
        "high_risk": "شحنات عالية الخطورة",
        "overview": "نظرة عامة على البيانات المستوردة",
        "preview": "معاينة التنبؤات صفا صفا",
        "weather_note": "النموذج المنشور لا يستعمل أي متغير للطقس (حسب مخطط الحزمة).",
        "decision_title": "دعم القرار التشغيلي",
        "decision_intro": "هامش ETA المقترح هو عدد الأيام التي يُنصح بإضافتها إلى الموعد الحالي لبلوغ مستوى الخدمة المختار.",
        "buffer_by_group": "هامش ETA المقترح حسب المسار وشركة النقل",
        "recommendations": "توصيات مبنية على البيانات",
        "download_decision": "تحميل جدول القرار (CSV)",
        "download_predictions": "تحميل التنبؤات (CSV)",
        "routes_at_risk": "أكثر المسارات حساسية",
        "carrier_risk": "متوسط الخطر حسب شركة النقل",
        "matrix_title": "مصفوفة الخطر × هامش ETA",
        "gap_dist_title": "توزيع فارق ETA (المتوقع - ETA)",
        "service_slider": "مستوى الخدمة (يعيد حساب الهامش مباشرة)",
        "preparation": "تتبع التحضير",
        "prep_initial": "الصفوف الأولية",
        "prep_removed": "الصفوف المحذوفة (غير المتسقة)",
        "prep_usable": "الصفوف القابلة للاستعمال",
        "prep_actual": "التاريخ الفعلي متوفر",
        "prep_weather": "متغيرات الطقس المستعملة في النموذج",
        "model_title": "النموذج المنشور والتفسير",
        "m_mae": "MAE (أيام)",
        "m_model": "النموذج",
        "m_rows": "صفوف التدريب",
        "m_r2": "R2 (اختبار)",
        "v_title": "تحقق متقدم",
        "v_temporal": "R2 الزمني",
        "v_auc": "AUC قاعدة الخطر",
        "v_falsealarm": "معدل الإنذارات الكاذبة",
        "v_baserate": "معدل التجاوز الفعلي",
        "importance_title": "أهمية المتغيرات (انخفاض R2 بالتبديل)",
        "model_honesty": "يؤكد التحقق الزمني استقرار توقع المدة؛ وكون AUC قاعدة الخطر قريبا من 0.5 يبيّن أن النموذج لا يميّز التأخيرات الفردية — القيمة في إعادة ضبط ETA وترتيب الأولويات.",
        "class_section": "صنف مدة التسليم (نموذج التصنيف)",
        "class_present": "تم تحميل نموذج صنف المدة",
        "class_absent": "لا يوجد نموذج لصنف المدة (تم تجاهل القسم).",
        "class_acc": "الدقة",
        "class_auc": "ROC-AUC",
        "class_f1": "F1 (ماكرو)",
        "class_long_share": "نسبة الصنف الطويل",
        "scatter3d_title": "عرض ثلاثي الأبعاد: المدة المتوقعة × ETA × درجة الخطر",
        "scatter3d_note": "كل نقطة شحنة؛ أدِر العرض لاستكشاف العلاقة بين المدة المتوقعة وETA ودرجة الخطر.",
        "class_dist_title": "توزيع أصناف المدة",
        "confusion_title": "مصفوفة الالتباس (اختبار مغلق)",
        "conf_pred": "متوقع", "conf_real": "فعلي",
        "class_no_model": "نموذج صنف المدة غير محمّل.",
        "assistant_title": "مساعد تفسير النتائج",
        "assistant_hint": "اسأل عن المسارات، شركات النقل، هوامش ETA أو درجة الأولوية.",
        "assistant_nature": "مساعد محلي حتمي قائم على قواعد: لا يستعمل أي ذكاء اصطناعي أو واجهة خارجية؛ يجيب فقط من نتائج الملف المرفوع.",
        "assistant_placeholder": "مثال: ما المسار الذي يجب مراقبته أولا؟",
        "low": "منخفض", "medium": "متوسط", "high": "مرتفع",
        "action_ok": "الإبقاء", "action_monitor": "مراقبة",
        "action_buffer": "إضافة هامش ETA", "action_escalate": "أولوية تشغيلية",
        "confidence_high": "قوية", "confidence_medium": "متوسطة", "confidence_low": "ضعيفة",
        "class_standard": "عادية", "class_long": "طويلة",
        "day_suffix": "ي",
        "col_route": "المسار", "col_carrier": "شركة النقل", "col_volume": "الحجم",
        "col_riskmean": "متوسط الخطر", "col_overrun": "نسبة تجاوز ETA",
        "col_buffer": "هامش ETA (ي)", "col_durmean": "متوسط المدة المتوقعة (ي)",
        "col_etamean": "متوسط ETA (ي)", "col_conf": "الثقة", "col_action": "الإجراء",
    },
}

ROLES = {
    "fr": ["date d'expedition", "ETA promise", "ville d'origine", "ville de destination",
           "transporteur", "fournisseur", "quantite", "cout d'expedition", "comparaison reel uniquement"],
    "en": ["shipping date", "promised ETA", "origin city", "destination city",
           "carrier", "supplier", "quantity", "shipping cost", "actual comparison only"],
    "ar": ["تاريخ الشحن", "ETA الموعود", "مدينة الانطلاق", "مدينة الوصول",
           "شركة النقل", "المورد", "الكمية", "تكلفة الشحن", "للمقارنة الفعلية فقط"],
}


def t(key: str):
    return TEXT[st.session_state.get("lang", "fr")][key]


def fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    return f"{x:.1%}"


def fmt_days(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    return f"{x:.1f} {t('day_suffix')}"


# ---------------------------------------------------------------------------
# CSS — raffine, deux tons (bleu ardoise + accent chaud), sans relief 3D
# ---------------------------------------------------------------------------
def inject_css(lang: str) -> None:
    rtl = lang == "ar"
    direction = "rtl" if rtl else "ltr"
    align = "right" if rtl else "left"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        .stApp {{
            direction: {direction};
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            background:
              radial-gradient(820px 380px at 6% -2%, rgba(31, 122, 140, .12), transparent 60%),
              radial-gradient(720px 360px at 96% 0%, rgba(232, 116, 59, .12), transparent 58%),
              linear-gradient(180deg, #f6f9fc 0%, #eef3f8 55%, #f7fafc 100%);
            color: {PALETTE['ink']};
        }}
        .block-container {{ padding-top: 1.0rem; max-width: 1360px; }}
        h1, h2, h3, h4 {{ color: {PALETTE['ink']} !important; text-align: {align}; letter-spacing: -.01em; }}
        p, div, label, span, li {{ text-align: {align}; }}
        /* Barre laterale : fond CLAIR, texte fonce, selecteurs blancs (lisibilite imperative) */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, #eef3f8 100%);
            border-{'left' if rtl else 'right'}: 1px solid #d8e1ea;
        }}
        section[data-testid="stSidebar"] * {{ color: {PALETTE['ink']} !important; }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stFileUploader label {{
            color: {PALETTE['slate']} !important; font-weight: 700 !important;
        }}
        /* Champs de selection / upload : fond blanc, texte fonce */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
            background: #ffffff !important; color: {PALETTE['ink']} !important;
            border: 1px solid #cdd8e3 !important; border-radius: 10px !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{ color: {PALETTE['ink']} !important; }}
        /* Menu deroulant (popover des selectbox) : fond blanc lisible */
        div[data-baseweb="popover"] li, ul[role="listbox"] li {{ color: {PALETTE['ink']} !important; }}

        /* En-tete (degrade, sans 3D) */
        .hero {{
            position: relative; overflow: hidden;
            border-radius: 18px; padding: 30px 34px; margin-bottom: 18px;
            background: linear-gradient(120deg, #16263d 0%, #1f3a5f 55%, #1f7a8c 120%);
            border: 1px solid rgba(255,255,255,.12);
            box-shadow: 0 18px 44px rgba(15, 27, 45, .26);
            animation: fadeIn .5s ease both;
        }}
        .hero::after {{
            content: ""; position: absolute; top: -80px; {'left' if rtl else 'right'}: -60px;
            width: 320px; height: 320px; border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(232,116,59,.55), rgba(244,162,97,.18) 55%, transparent 70%);
            filter: blur(4px); opacity: .9;
        }}
        .hero h1 {{ margin: 0 0 4px; font-size: 2.1rem; font-weight: 800; color: #fff !important; }}
        .hero .sub {{ color: #cfe2ec; font-size: .95rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; }}
        .hero p {{ color: #e6eef5; max-width: 980px; margin: 12px 0 0; line-height: 1.65; font-size: 1.02rem; position: relative; z-index: 2; }}
        .hero .pills {{ margin-top: 14px; position: relative; z-index: 2; }}
        .pill {{
            display: inline-block; margin-{'left' if rtl else 'right'}: 8px; margin-top: 8px;
            background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
            color: #fff; padding: 5px 12px; border-radius: 999px; font-size: .82rem; font-weight: 600;
        }}

        /* Bandeau honnetete */
        .honest {{
            border-{'right' if rtl else 'left'}: 4px solid {PALETTE['accent']};
            background: #fff7f2; color: #7c3a16; border-radius: 10px;
            padding: 12px 16px; margin: 6px 0 16px; font-weight: 600;
            box-shadow: 0 8px 22px rgba(232,116,59,.08);
        }}

        /* Cartes */
        .metric-card, .info-card, .rec-card {{
            border-radius: 14px; background: #ffffff; border: 1px solid #e2e9f1;
            box-shadow: 0 12px 30px rgba(15, 27, 45, .07);
            padding: 16px 18px; height: 100%;
            transition: transform .16s ease, box-shadow .16s ease;
            animation: fadeIn .5s ease both;
        }}
        .metric-card:hover, .info-card:hover, .rec-card:hover {{
            transform: translateY(-5px) scale(1.012);
            box-shadow: 0 26px 50px rgba(15, 27, 45, .18), 0 4px 10px rgba(15,27,45,.10);
        }}
        .metric-value {{ font-size: 1.7rem; font-weight: 800; color: {PALETTE['teal']}; }}
        .metric-label {{ font-size: .76rem; color: {PALETTE['muted']}; text-transform: uppercase; letter-spacing: .06em; margin-top: 2px; }}
        .info-card h4, .rec-card h4 {{ margin: 0 0 8px; font-size: 1.02rem; color: {PALETTE['slate']}; }}
        .info-card p, .rec-card p {{ margin: 0; color: #475569; line-height: 1.6; }}
        .info-card .ic {{ font-size: 1.5rem; }}
        .rec-card h4 {{
            width: 30px; height: 30px; line-height: 30px; text-align: center;
            border-radius: 8px; background: {PALETTE['accent']}; color: #fff !important;
        }}
        .chart-note {{
            border-{'right' if rtl else 'left'}: 4px solid {PALETTE['teal']};
            padding: 9px 13px; background: #ecf6f8; color: #134e57;
            border-radius: 8px; margin: 6px 0 12px; font-size: .9rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px 10px 0 0; background: rgba(255,255,255,.6);
            border: 1px solid #e2e9f1; padding: 9px 16px; font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{ background: #fff; color: {PALETTE['teal']} !important; }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid #e2e9f1; border-radius: 12px; overflow: hidden;
            box-shadow: 0 10px 26px rgba(15,27,45,.05);
        }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def info_card(icon: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="info-card"><div class="ic">{icon}</div><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def chart_note(text: str) -> None:
    st.markdown(f'<div class="chart-note">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chargements (caches). Aucun reentrainement.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


@st.cache_resource(show_spinner=False)
def load_class_bundle():
    """Bundle de classification de la duree : charge si present, ignore sinon."""
    if not CLASS_BUNDLE_PATH.exists():
        return None
    try:
        return joblib.load(CLASS_BUNDLE_PATH)
    except Exception:
        return None  # ex : lib (xgboost/lightgbm) absente -> on ignore sans erreur


@st.cache_data(show_spinner=False)
def load_report() -> dict:
    if not REPORT_PATH.exists():
        return {}
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    """Metriques academiques (dont la classification) ecrites par le notebook."""
    p = HERE / "ml_academique" / "metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Pipeline (identique au notebook), inline -> app auto-contenu
# ---------------------------------------------------------------------------
ALIASES = {
    "shipment_date": "ship_date", "shipping_date": "ship_date", "date_expedition": "ship_date",
    "eta": "expected_delivery_date", "promised_date": "expected_delivery_date",
    "expected_date": "expected_delivery_date",
    "origin": "origin_city", "destination": "destination_city",
    "carrier": "logistics_company", "transporter": "logistics_company", "transporteur": "logistics_company",
    "supplier": "supplier_name", "fournisseur": "supplier_name",
    "qty": "quantity", "shipping_cost": "shipping_cost_ngn", "cost": "shipping_cost_ngn",
    "actual_date": "actual_delivery_date", "delivered_date": "actual_delivery_date",
    "status": "delivery_status",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        rename[col] = ALIASES.get(key, key)
    return df.rename(columns=rename)


def read_uploaded(uploaded) -> pd.DataFrame:
    """Detecte le format par l'extension, avec replis robustes. Ne plante jamais."""
    raw = uploaded.read()
    name = uploaded.name.lower()
    buf = io.BytesIO(raw)
    readers = []
    if name.endswith((".parquet", ".pq")):
        readers.append(lambda b: pd.read_parquet(b))
    if name.endswith((".xlsx", ".xls")):
        readers.append(lambda b: pd.read_excel(b))
    if name.endswith(".json"):
        readers.append(lambda b: pd.read_json(b))
    if name.endswith((".csv", ".txt", ".tsv")):
        for sep in [",", ";", "\t", "|"]:
            readers.append(lambda b, sep=sep: pd.read_csv(b, sep=sep))
    # Replis pour extension absente / erronee
    readers.extend([
        lambda b: pd.read_parquet(b), lambda b: pd.read_excel(b),
        lambda b: pd.read_json(b), lambda b: pd.read_csv(b),
        lambda b: pd.read_csv(b, sep=";"), lambda b: pd.read_csv(b, sep="\t"),
    ])
    last_error = None
    for reader in readers:
        try:
            buf.seek(0)
            df = reader(buf)
            if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
                return normalize_columns(df)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"{t('format_error')} ({last_error})")


def engineer_features(df_raw: pd.DataFrame):
    """Construit les variables connues AVANT expedition + cibles si actual dispo."""
    report = {"initial": int(len(df_raw))}
    df = df_raw.copy()
    for c in ["ship_date", "expected_delivery_date", "actual_delivery_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["estimated_transit_days"] = (df["expected_delivery_date"] - df["ship_date"]).dt.days
    df["ship_month"] = df["ship_date"].dt.month.astype("Int64").astype(str)
    df["ship_day_of_week"] = df["ship_date"].dt.dayofweek.astype("Int64").astype(str)

    has_actual = "actual_delivery_date" in df.columns and df["actual_delivery_date"].notna().any()
    if has_actual:
        df["realized_transit_days"] = (df["actual_delivery_date"] - df["ship_date"]).dt.days
        df["delay_days"] = (df["actual_delivery_date"] - df["expected_delivery_date"]).dt.days.clip(lower=0)

    before = len(df)
    df = df.dropna(subset=["estimated_transit_days", "ship_month", "ship_day_of_week"]).copy()
    df = df[df["estimated_transit_days"] >= 0].copy()
    report["removed"] = int(before - len(df))
    report["usable"] = int(len(df))
    report["has_actual"] = bool(has_actual)
    return df, report


def build_X(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Matrice d'entree depuis le SCHEMA du bundle (le ColumnTransformer cible par nom)."""
    cols = list(schema.get("numeric", [])) + list(schema.get("one_hot", [])) + list(schema.get("target_encoded", []))
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].copy()


# ---------------------------------------------------------------------------
# Score de risque, decisions, recommandations, assistant
# ---------------------------------------------------------------------------
def risk_score_from_margin(margin: pd.Series, mae_days: float) -> pd.Series:
    scale = max(float(mae_days) / 1.7, 0.75)
    return 1 / (1 + np.exp(-(margin / scale)))


def risk_level(score: float) -> str:
    if score >= 0.70:
        return t("high")
    if score >= 0.45:
        return t("medium")
    return t("low")


def action_label(buffer_days: float, avg_risk: float) -> str:
    if buffer_days >= 3 or avg_risk >= 0.72:
        return t("action_escalate")
    if buffer_days >= 1:
        return t("action_buffer")
    if avg_risk >= 0.45:
        return t("action_monitor")
    return t("action_ok")


def confidence_label(volume: int) -> str:
    if volume >= 80:
        return t("confidence_high")
    if volume >= 25:
        return t("confidence_medium")
    return t("confidence_low")


def service_quantile(key: str) -> float:
    return {"balanced": 0.80, "reliable": 0.90, "premium": 0.95}[key]


def prepare_predictions(df_raw: pd.DataFrame, bundle: dict, class_bundle, mae_days: float):
    df, prep = engineer_features(df_raw)
    if len(df) == 0:
        return df, prep

    X = build_X(df, bundle["feature_schema"])
    pred = np.maximum(bundle["pipeline"].predict(X), 0)

    res = df.copy()
    res["route"] = res["origin_city"].astype(str) + " → " + res["destination_city"].astype(str)
    res["predicted_duration_days"] = pred
    res["eta_gap_days"] = res["predicted_duration_days"] - res["estimated_transit_days"].astype(float)
    res["predicted_delay_days"] = res["eta_gap_days"].clip(lower=0)
    res["risk_score"] = risk_score_from_margin(res["eta_gap_days"], mae_days)
    res["risk_level"] = res["risk_score"].apply(risk_level)
    res["predicted_eta_overrun"] = res["eta_gap_days"] > 0

    # Classe de duree (optionnelle)
    if class_bundle is not None:
        try:
            Xc = build_X(df, class_bundle["feature_schema"])
            pipe = class_bundle["pipeline"]
            pred_c = pipe.predict(Xc)
            proba_c = pipe.predict_proba(Xc)
            res["predicted_duration_class"] = [
                t("class_long") if int(c) == 1 else t("class_standard") for c in pred_c
            ]
            res["class_confidence"] = proba_c.max(axis=1)
        except Exception:
            pass

    if "actual_delivery_date" in res.columns and res["actual_delivery_date"].notna().any():
        res["actual_duration_days"] = (res["actual_delivery_date"] - res["ship_date"]).dt.days

    return res, prep


def decision_table(res: pd.DataFrame, quantile: float) -> pd.DataFrame:
    rows = []
    for (route, carrier), g in res.groupby(["route", "logistics_company"], dropna=False):
        buffer = int(math.ceil(max(0.0, float(g["eta_gap_days"].quantile(quantile)))))
        avg_risk = float(g["risk_score"].mean())
        rows.append({
            "route": route, "logistics_company": carrier, "volume": int(len(g)),
            "risk_mean": avg_risk,
            "predicted_overrun_rate": float(g["predicted_eta_overrun"].mean()),
            "eta_buffer_recommended_days": buffer,
            "predicted_duration_mean": float(g["predicted_duration_days"].mean()),
            "planned_eta_mean": float(g["estimated_transit_days"].mean()),
            "priority_score": float(avg_risk * 100 + buffer * 9 + min(len(g), 100) / 4),
            "confidence": confidence_label(len(g)),
            "decision": action_label(buffer, avg_risk),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["priority_score", "volume"], ascending=False).reset_index(drop=True)


def recommendations(decisions: pd.DataFrame, res: pd.DataFrame, lang: str) -> list:
    if decisions.empty:
        return []
    top = decisions.iloc[0]
    risky = decisions[decisions["eta_buffer_recommended_days"] >= 1].head(3)
    grp = res.groupby("logistics_company")["risk_score"].agg(["mean", "size"]).query("size >= 5")
    high_carrier = grp.sort_values("mean", ascending=False)
    best_carrier = grp.sort_values("mean", ascending=True)
    q75 = fmt_pct(decisions["predicted_overrun_rate"].quantile(.75))
    tb = int(top["eta_buffer_recommended_days"])

    if lang == "en":
        recs = [
            f"Add a {tb}-day ETA buffer on {top['route']} with {top['logistics_company']} first; it has the highest operational priority.",
            f"Monitor lanes whose predicted overrun rate exceeds {q75}; trigger proactive customer communication.",
        ]
        if len(high_carrier) and len(best_carrier):
            recs.append(f"Carrier allocation: {high_carrier.index[0]} is the riskiest here, while {best_carrier.index[0]} is the most stable (sufficient volume).")
        if not risky.empty:
            recs.append(f"Start with {len(risky)} lane-carrier groups needing at least a one-day ETA buffer.")
        return recs
    if lang == "ar":
        recs = [
            f"أضف هامش ETA قدره {tb} يوم على المسار {top['route']} مع {top['logistics_company']} أولا لأنه الأعلى أولوية.",
            f"راقب المسارات التي يتجاوز فيها معدل التجاوز المتوقع {q75}، مع تنبيه استباقي للعميل.",
        ]
        if len(high_carrier) and len(best_carrier):
            recs.append(f"توزيع الناقلين: {high_carrier.index[0]} الأكثر خطرا هنا، بينما {best_carrier.index[0]} الأكثر استقرارا (حجم كافٍ).")
        if not risky.empty:
            recs.append(f"ابدأ بـ {len(risky)} مجموعات مسار/ناقل تحتاج إلى هامش ETA لا يقل عن يوم.")
        return recs
    recs = [
        f"Ajouter d'abord un tampon ETA de {tb} jour(s) sur {top['route']} avec {top['logistics_company']} : priorite operationnelle la plus elevee.",
        f"Surveiller les axes dont le taux de depassement predit depasse {q75} et declencher une communication client proactive.",
    ]
    if len(high_carrier) and len(best_carrier):
        recs.append(f"Allocation transporteur : {high_carrier.index[0]} est le plus risque ici, {best_carrier.index[0]} le plus stable (volume suffisant).")
    if not risky.empty:
        recs.append(f"Commencer par {len(risky)} groupes axe/transporteur exigeant au moins un jour de tampon ETA.")
    return recs


def assistant_answer(question: str, decisions: pd.DataFrame, res: pd.DataFrame, lang: str) -> str:
    q = (question or "").lower()
    if decisions.empty:
        return {"fr": "Pas assez de donnees pour repondre.",
                "en": "Not enough data to answer yet.",
                "ar": "لا توجد بيانات كافية للإجابة."}[lang]
    top = decisions.iloc[0]
    carriers = res.groupby("logistics_company")["risk_score"].mean().sort_values()
    overrun = float(res["predicted_eta_overrun"].mean())
    avg_risk = float(res["risk_score"].mean())
    avg_gap = float(res["eta_gap_days"].mean())
    tb = int(top["eta_buffer_recommended_days"])

    wants_carrier = any(w in q for w in ["transport", "carrier", "شركة", "ناقل"])
    wants_buffer = any(w in q for w in ["tampon", "buffer", "هامش"])
    wants_route = any(w in q for w in ["axe", "route", "lane", "مسار"])
    wants_score = any(w in q for w in ["score", "risque", "risk", "priorisation", "priorit", "خطر", "أولوية"])

    if lang == "en":
        gl = ("ETA = the date promised to the customer. The ETA buffer is the extra days to add when "
              "predicted duration may exceed the promise. The risk score is a 0-100% prioritization "
              "indicator, not a calibrated probability of individual delay.")
        if wants_carrier and len(carriers):
            return f"{gl}\n\nMost stable carrier: {carriers.index[0]} ({fmt_pct(carriers.iloc[0])}). Riskiest: {carriers.index[-1]} ({fmt_pct(carriers.iloc[-1])}). Add monitoring/ETA buffer on the riskiest, especially high-volume lanes."
        if wants_buffer:
            return f"{gl}\n\nFirst buffer: add {tb} day(s) on {top['route']} with {top['logistics_company']} ({fmt_pct(top['risk_mean'])} avg risk, {fmt_pct(top['predicted_overrun_rate'])} predicted overrun, {int(top['volume'])} shipments)."
        if wants_route:
            return f"{gl}\n\nLane to monitor first: {top['route']} with {top['logistics_company']} — {fmt_pct(top['risk_mean'])} avg risk, {fmt_pct(top['predicted_overrun_rate'])} predicted overrun, {tb}-day buffer. Action: {top['decision']}."
        if wants_score:
            return f"{gl}\n\nIt ranks lane-carrier groups so you act on the most exposed first; it is not a guarantee for any single parcel."
        return f"{gl}\n\nSummary: {fmt_pct(overrun)} of shipments may exceed ETA, average risk {fmt_pct(avg_risk)}, average gap {avg_gap:.2f} d. Start with {top['route']} / {top['logistics_company']} (+{tb} d buffer)."
    if lang == "ar":
        gl = ("ETA هو التاريخ الموعود للعميل. هامش ETA هو الأيام الإضافية عندما قد تتجاوز المدة المتوقعة الموعد. "
              "درجة الخطر مؤشر ترتيب أولويات بين 0% و100%، وليست احتمالا معايرا لتأخير فردي.")
        if wants_carrier and len(carriers):
            return f"{gl}\n\nالأكثر استقرارا: {carriers.index[0]} ({fmt_pct(carriers.iloc[0])}). الأكثر خطرا: {carriers.index[-1]} ({fmt_pct(carriers.iloc[-1])}). أضف مراقبة أو هامش ETA للأكثر خطرا خاصة في المسارات الكبيرة."
        if wants_buffer:
            return f"{gl}\n\nأول هامش: أضف {tb} يوم على {top['route']} مع {top['logistics_company']} (متوسط خطر {fmt_pct(top['risk_mean'])}، تجاوز متوقع {fmt_pct(top['predicted_overrun_rate'])}، {int(top['volume'])} شحنة)."
        if wants_route:
            return f"{gl}\n\nالمسار الأول للمراقبة: {top['route']} مع {top['logistics_company']} — متوسط خطر {fmt_pct(top['risk_mean'])}، تجاوز متوقع {fmt_pct(top['predicted_overrun_rate'])}، هامش {tb} يوم. القرار: {top['decision']}."
        if wants_score:
            return f"{gl}\n\nيرتّب مجموعات المسار/الناقل لتتحرك أولا نحو الأكثر تعرضا؛ وليس ضمانا لأي شحنة بمفردها."
        return f"{gl}\n\nالخلاصة: {fmt_pct(overrun)} من الشحنات قد تتجاوز ETA، متوسط الخطر {fmt_pct(avg_risk)}، متوسط الفجوة {avg_gap:.2f} ي. ابدأ بـ {top['route']} / {top['logistics_company']} (+{tb} ي)."

    gl = ("ETA = la date promise au client. Le tampon ETA est le nombre de jours a ajouter quand la duree "
          "predite peut depasser la promesse. Le score de risque est un indicateur de priorisation 0-100 %, "
          "pas une probabilite calibree de retard individuel.")
    if wants_carrier and len(carriers):
        return f"{gl}\n\nTransporteur le plus stable : {carriers.index[0]} ({fmt_pct(carriers.iloc[0])}). Le plus risque : {carriers.index[-1]} ({fmt_pct(carriers.iloc[-1])}). Surveiller / ajouter un tampon ETA sur le plus risque, surtout a fort volume."
    if wants_buffer:
        return f"{gl}\n\nPremier tampon : {tb} jour(s) sur {top['route']} avec {top['logistics_company']} ({fmt_pct(top['risk_mean'])} de risque moyen, {fmt_pct(top['predicted_overrun_rate'])} de depassement predit, {int(top['volume'])} expeditions)."
    if wants_route:
        return f"{gl}\n\nAxe a surveiller en premier : {top['route']} avec {top['logistics_company']} — risque moyen {fmt_pct(top['risk_mean'])}, depassement predit {fmt_pct(top['predicted_overrun_rate'])}, tampon {tb} jour(s). Decision : {top['decision']}."
    if wants_score:
        return f"{gl}\n\nIl classe les groupes axe/transporteur pour agir d'abord sur les plus exposes ; ce n'est pas une garantie pour un colis isole."
    return f"{gl}\n\nSynthese : {fmt_pct(overrun)} des expeditions risquent de depasser l'ETA, risque moyen {fmt_pct(avg_risk)}, marge moyenne {avg_gap:.2f} j. Priorite : {top['route']} / {top['logistics_company']} (+{tb} j)."


@st.cache_data(show_spinner=False)
def example_csv() -> bytes:
    """Petit fichier-exemple au bon schema (genere depuis le parquet si dispo)."""
    cols = REQUIRED_RAW + ["actual_delivery_date"]
    if DATA_PATH.exists():
        try:
            d = pd.read_parquet(DATA_PATH)
            keep = [c for c in cols if c in d.columns]
            return d[keep].head(40).to_csv(index=False).encode("utf-8-sig")
        except Exception:
            pass
    sample = pd.DataFrame({
        "ship_date": ["2024-03-01", "2024-03-02"],
        "expected_delivery_date": ["2024-03-08", "2024-03-07"],
        "origin_city": ["Lagos", "Abuja"],
        "destination_city": ["Kano", "Ibadan"],
        "logistics_company": ["DHL Nigeria", "GIG Logistics"],
        "supplier_name": ["Supplier_A", "Supplier_B"],
        "quantity": [3, 1],
        "shipping_cost_ngn": [12000, 5400],
        "actual_delivery_date": ["2024-03-10", "2024-03-07"],
    })
    return sample.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------------------
st.session_state.setdefault("lang", "fr")

with st.sidebar:
    lang_label = {"fr": "Français", "en": "English", "ar": "العربية"}
    chosen = st.selectbox(
        TEXT[st.session_state["lang"]]["language"],
        list(lang_label.values()),
        index=list(lang_label.keys()).index(st.session_state["lang"]),
    )
    st.session_state["lang"] = {v: k for k, v in lang_label.items()}[chosen]
    lang = st.session_state["lang"]
    service_labels = {t("service_balanced"): "balanced", t("service_reliable"): "reliable", t("service_premium"): "premium"}
    service_choice = st.selectbox(t("service_level"), list(service_labels.keys()), index=1)
    uploaded = st.file_uploader(t("upload"), type=None, help=t("upload_help"))

lang = st.session_state["lang"]
inject_css(lang)

bundle = load_bundle()
class_bundle = load_class_bundle()
report = load_report()
meta = load_meta()
mae_days = float(report.get("test_MAE", 2.0))

# En-tete
st.markdown(
    f"""
    <div class="hero">
      <div class="sub">{t('brand')} · {t('tagline')}</div>
      <h1>{t('hero_title')}</h1>
      <p>{t('hero_body')}</p>
      <div class="pills"><span class="pill">ETA</span><span class="pill">risk score</span>
      <span class="pill">ETA buffer</span><span class="pill">FR · EN · AR</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(f'<div class="honest">⚠️ {t("honest_phrase")}</div>', unsafe_allow_html=True)

schema_df = pd.DataFrame({
    t("schema_col"): REQUIRED_RAW + ["actual_delivery_date"],
    t("schema_required"): [t("schema_yes")] * len(REQUIRED_RAW) + [t("schema_optional")],
    t("schema_role"): ROLES[lang],
})

# --- Ecran d'accueil (avant tout upload) ---
demo_mode = st.session_state.get("demo_mode", False)
if uploaded is None and not demo_mode:
    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("🧭", t("home_what_title"), t("home_what_body"))
    with c2:
        info_card("📄", t("home_format_title"),
                  f"{t('home_format_body')} <b>{', '.join(REQUIRED_RAW)}</b>.<br><br>"
                  f"{t('home_format_optional')}<br><br>{t('home_format_files')}")
    with c3:
        info_card("🎯", t("home_limits_title"), t("home_limits_body"))

    st.write("")
    h1, h2 = st.columns(2)
    with h1:
        st.download_button("⬇️ " + t("example_button"), example_csv(),
                           "logipredict_exemple.csv", "text/csv", use_container_width=True)
    with h2:
        if DATA_PATH.exists() and st.button("▶️ " + t("demo_button"), use_container_width=True):
            st.session_state["demo_mode"] = True
            st.rerun()
    st.info(t("waiting"))
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
    st.stop()

if bundle is None:
    st.error(t("model_missing"))
    st.stop()

# --- Lecture + validation du fichier (upload ou demonstration) ---
try:
    if uploaded is not None:
        df_raw = read_uploaded(uploaded)
    else:  # mode demonstration : echantillon du parquet de reference
        df_raw = normalize_columns(pd.read_parquet(DATA_PATH).head(4000).copy())
except Exception as exc:
    st.error(str(exc))
    st.download_button("⬇️ " + t("example_button"), example_csv(), "logipredict_exemple.csv", "text/csv")
    st.stop()

missing = [c for c in REQUIRED_RAW if c not in df_raw.columns]
if missing:
    st.error(f"{t('missing_cols')} : {', '.join(missing)}")
    st.caption(f"{t('found_cols')} : {', '.join(map(str, df_raw.columns))}")
    st.download_button("⬇️ " + t("example_button"), example_csv(), "logipredict_exemple.csv", "text/csv")
    st.stop()

with st.spinner(t("preparation")):
    res, prep = prepare_predictions(df_raw, bundle, class_bundle, mae_days)

if res.empty:
    st.error(t("no_usable"))
    st.stop()

quantile = service_quantile(service_labels[service_choice])
decisions = decision_table(res, quantile)
recs = recommendations(decisions, res, lang)

tabs = st.tabs(t("tabs"))

# ---- Onglet 1 : Vue generale ----
with tabs[0]:
    st.subheader(t("overview"))
    has_class = "predicted_duration_class" in res.columns
    long_label = t("class_long")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: metric_card(t("raw_rows"), f"{len(df_raw):,}")
    with k2: metric_card(t("usable_rows"), f"{len(res):,}")
    with k3: metric_card(t("predicted_late"), fmt_pct(res["predicted_eta_overrun"].mean()))
    with k4: metric_card(t("avg_risk"), fmt_pct(res["risk_score"].mean()))
    with k5:
        long_share = float((res["predicted_duration_class"] == long_label).mean()) if has_class else None
        metric_card(t("class_long_share"), fmt_pct(long_share) if long_share is not None else "—")
    st.caption(t("weather_note"))
    st.markdown(f"#### {t('preview')}")
    preview_cols = [c for c in [
        "ship_date", "expected_delivery_date", "route", "logistics_company", "supplier_name",
        "quantity", "shipping_cost_ngn", "estimated_transit_days", "predicted_duration_days",
        "predicted_duration_class", "class_confidence", "eta_gap_days", "predicted_delay_days",
        "risk_score", "risk_level",
    ] if c in res.columns]
    st.dataframe(res[preview_cols].head(100), use_container_width=True, hide_index=True)

# ---- Onglet 2 : Aide a la decision ----
with tabs[1]:
    st.subheader(t("decision_title"))
    st.caption(t("decision_intro"))
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(t("avg_buffer"), fmt_days(decisions["eta_buffer_recommended_days"].mean() if len(decisions) else 0))
    with c2: metric_card(t("high_risk"), f"{int((res['risk_score'] >= 0.70).sum()):,}")
    with c3: metric_card(t("predicted_late"), fmt_pct(res["predicted_eta_overrun"].mean()))
    with c4: metric_card(t("avg_risk"), fmt_pct(res["risk_score"].mean()))

    if recs:
        st.markdown(f"#### {t('recommendations')}")
        rcols = st.columns(min(2, len(recs)))
        for i, rec in enumerate(recs):
            with rcols[i % len(rcols)]:
                st.markdown(f'<div class="rec-card"><h4>{i+1}</h4><p>{rec}</p></div>', unsafe_allow_html=True)

    st.markdown(f"#### {t('buffer_by_group')}")
    show = decisions.rename(columns={
        "route": t("col_route"), "logistics_company": t("col_carrier"), "volume": t("col_volume"),
        "risk_mean": t("col_riskmean"), "predicted_overrun_rate": t("col_overrun"),
        "eta_buffer_recommended_days": t("col_buffer"), "predicted_duration_mean": t("col_durmean"),
        "planned_eta_mean": t("col_etamean"), "confidence": t("col_conf"), "decision": t("col_action"),
    }).drop(columns=["priority_score"])
    st.dataframe(show.head(150), use_container_width=True, hide_index=True)
    st.download_button(t("download_decision"), show.to_csv(index=False).encode("utf-8-sig"),
                       "logipredict_decision_table.csv", "text/csv")

# ---- Onglet 3 : Risques & axes ----
with tabs[2]:
    serv = st.slider(t("service_slider"), 50, 99, int(quantile * 100), 1)
    live_q = serv / 100.0
    live_dec = decision_table(res, live_q)

    left, right = st.columns(2)
    with left:
        st.markdown(f"#### {t('routes_at_risk')}")
        routes = (res.groupby("route")
                  .agg(volume=("route", "size"), risk_mean=("risk_score", "mean"),
                       overrun=("predicted_eta_overrun", "mean"))
                  .query("volume >= 3").sort_values("risk_mean", ascending=False).head(15).reset_index())
        if not routes.empty:
            fig = px.bar(routes, x="risk_mean", y="route", orientation="h", color="overrun",
                         color_continuous_scale=[PALETTE["low"], PALETTE["medium"], PALETTE["high"]],
                         text=routes["risk_mean"].map(lambda x: f"{x:.0%}"))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(height=460, xaxis_tickformat=".0%",
                              yaxis={"categoryorder": "total ascending"},
                              margin=dict(l=8, r=30, t=10, b=8))
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown(f"#### {t('carrier_risk')}")
        carr = (res.groupby("logistics_company")
                .agg(volume=("logistics_company", "size"), risk_mean=("risk_score", "mean"),
                     overrun=("predicted_eta_overrun", "mean"))
                .sort_values("risk_mean", ascending=False).reset_index())
        fig = px.bar(carr, x="risk_mean", y="logistics_company", orientation="h", color="overrun",
                     color_continuous_scale=[PALETTE["low"], PALETTE["medium"], PALETTE["high"]],
                     text=carr["risk_mean"].map(lambda x: f"{x:.0%}"))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=460, xaxis_tickformat=".0%",
                          yaxis={"categoryorder": "total ascending"},
                          margin=dict(l=8, r=30, t=10, b=8))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"#### {t('matrix_title')}")
    mat = live_dec.head(30).copy()
    if not mat.empty:
        mat["axe_transporteur"] = mat["route"].astype(str) + " | " + mat["logistics_company"].astype(str)
        fig = px.scatter(mat, x="risk_mean", y="eta_buffer_recommended_days", size="volume",
                         color="priority_score", hover_data=["axe_transporteur", "predicted_overrun_rate", "confidence", "decision"],
                         color_continuous_scale=[PALETTE["low"], PALETTE["medium"], PALETTE["high"]])
        fig.update_layout(height=420, xaxis_tickformat=".0%", margin=dict(l=8, r=8, t=10, b=8))
        st.plotly_chart(fig, use_container_width=True)

    cgap, cclass = st.columns([3, 2])
    with cgap:
        st.markdown(f"#### {t('gap_dist_title')}")
        fig = px.histogram(res, x="eta_gap_days", nbins=30, color="predicted_eta_overrun",
                           color_discrete_map={False: PALETTE["low"], True: PALETTE["high"]})
        fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["ink"])
        fig.update_layout(height=360, bargap=.05, margin=dict(l=8, r=8, t=10, b=8))
        st.plotly_chart(fig, use_container_width=True)
    with cclass:
        if "predicted_duration_class" in res.columns:
            st.markdown(f"#### {t('class_dist_title')}")
            dist = res["predicted_duration_class"].value_counts().reset_index()
            dist.columns = ["classe", "n"]
            figc = px.pie(dist, names="classe", values="n", hole=.55,
                          color="classe",
                          color_discrete_map={t("class_standard"): PALETTE["teal"], t("class_long"): PALETTE["accent"]})
            figc.update_layout(height=360, margin=dict(l=8, r=8, t=10, b=8),
                               legend=dict(orientation="h", y=-.05))
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.markdown(f"#### {t('class_dist_title')}")
            st.caption(t("class_no_model"))

    # Touche 3D legere et pertinente : duree predite x ETA x score de risque
    st.markdown(f"#### {t('scatter3d_title')}")
    chart_note(t("scatter3d_note"))
    s3d = res.sample(min(1500, len(res)), random_state=42)
    fig3d = px.scatter_3d(
        s3d, x="estimated_transit_days", y="predicted_duration_days", z="risk_score",
        color="risk_score",
        color_continuous_scale=[PALETTE["low"], PALETTE["medium"], PALETTE["high"]],
        hover_data=["route", "logistics_company"],
        labels={"estimated_transit_days": "ETA (j)", "predicted_duration_days": t("col_durmean"),
                "risk_score": t("avg_risk")})
    fig3d.update_traces(marker=dict(size=3.2, opacity=.75))
    fig3d.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                        scene=dict(xaxis_title="ETA (j)", yaxis_title="duree predite (j)", zaxis_title="risk"))
    st.plotly_chart(fig3d, use_container_width=True)

# ---- Onglet 4 : Preparation ----
with tabs[3]:
    st.subheader(t("preparation"))
    n_weather = len(bundle.get("weather_features", []) or [])
    prep_df = pd.DataFrame([
        (t("prep_initial"), f"{prep.get('initial', 0):,}"),
        (t("prep_removed"), f"{prep.get('removed', 0):,}"),
        (t("prep_usable"), f"{prep.get('usable', 0):,}"),
        (t("prep_actual"), t("schema_yes") if prep.get("has_actual") else "—"),
        (t("prep_weather"), str(n_weather)),
    ], columns=[" ", "  "])
    st.dataframe(prep_df, use_container_width=True, hide_index=True)
    st.caption(t("weather_note"))

# ---- Onglet 5 : Modele & interpretation ----
with tabs[4]:
    st.subheader(t("model_title"))
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card(t("m_mae"), f"{mae_days:.2f}")
    with m2: metric_card(t("m_model"), str(bundle.get("model_name", report.get("model", "HistGradientBoosting"))))
    with m3: metric_card(t("m_rows"), f"{int(report.get('n_train', 0)):,}")
    with m4: metric_card(t("m_r2"), f"{float(report.get('test_R2', 0)):.3f}")

    temporal = report.get("temporal_validation", {})
    risk_rule = report.get("risk_rule_metrics", {})
    if temporal or risk_rule:
        st.markdown(f"#### {t('v_title')}")
        v1, v2, v3, v4 = st.columns(4)
        with v1: metric_card(t("v_temporal"), f"{float(temporal.get('R2', 0)):.3f}" if temporal else "n/a")
        with v2: metric_card(t("v_auc"), f"{float(risk_rule.get('roc_auc', 0)):.3f}" if risk_rule else "n/a")
        with v3: metric_card(t("v_falsealarm"), fmt_pct(float(risk_rule.get("default_false_alarm_rate", 0))) if risk_rule else "n/a")
        with v4: metric_card(t("v_baserate"), fmt_pct(float(risk_rule.get("base_rate", 0))) if risk_rule else "n/a")

    st.warning(t("model_honesty"))

    importance = report.get("permutation_importance", [])
    if importance:
        imp = pd.DataFrame(importance).head(12).sort_values("chute_R2")
        st.markdown(f"#### {t('importance_title')}")
        fig = px.bar(imp, x="chute_R2", y="variable", orientation="h", color="chute_R2",
                     color_continuous_scale=[PALETTE["high"], "#f1f5f9", PALETTE["low"]])
        fig.update_layout(height=440, margin=dict(l=8, r=8, t=10, b=8))
        st.plotly_chart(fig, use_container_width=True)

    # Section classe de duree (optionnelle) — metriques du test scelle (notebook / metadata)
    st.markdown(f"#### {t('class_section')}")
    if class_bundle is not None:
        st.success(f"{t('class_present')} — {class_bundle.get('model_name', '?')} "
                   f"(seuil = {class_bundle.get('seuil_jours', '?')} j)")
        cm_acc = meta.get("clf_duree_accuracy")
        cm_auc = meta.get("clf_duree_auc_roc")
        cm_pr = meta.get("clf_duree_pr_auc")
        cc1, cc2, cc3 = st.columns(3)
        with cc1: metric_card(t("class_auc"), f"{float(cm_auc):.3f}" if cm_auc is not None else "—")
        with cc2: metric_card("PR-AUC", f"{float(cm_pr):.3f}" if cm_pr is not None else "—")
        with cc3: metric_card(t("class_acc"), f"{float(cm_acc):.3f}" if cm_acc is not None else "—")

        conf = meta.get("clf_duree_confusion")
        if conf:
            import numpy as _np
            cm = _np.array([[conf.get("tn", 0), conf.get("fp", 0)],
                            [conf.get("fn", 0), conf.get("tp", 0)]])
            labels = [t("class_standard"), t("class_long")]
            figcm = px.imshow(cm, x=labels, y=labels, text_auto=True,
                              color_continuous_scale="Blues",
                              labels=dict(x=t("conf_pred"), y=t("conf_real"), color="n"))
            figcm.update_layout(height=360, margin=dict(l=8, r=8, t=30, b=8),
                                title=t("confusion_title"), coloraxis_showscale=False)
            st.plotly_chart(figcm, use_container_width=True)
    else:
        st.caption(t("class_absent"))

    exp_cols = [c for c in [
        "ship_date", "expected_delivery_date", "route", "logistics_company", "supplier_name",
        "quantity", "shipping_cost_ngn", "estimated_transit_days", "predicted_duration_days",
        "predicted_duration_class", "class_confidence", "eta_gap_days", "predicted_delay_days",
        "risk_score", "risk_level", "predicted_eta_overrun",
    ] if c in res.columns]
    st.download_button(t("download_predictions"), res[exp_cols].to_csv(index=False).encode("utf-8-sig"),
                       "logipredict_predictions.csv", "text/csv")

# ---- Onglet 6 : Assistant ----
with tabs[5]:
    st.subheader(t("assistant_title"))
    st.caption(t("assistant_hint"))
    st.info(t("assistant_nature"))
    st.session_state.setdefault("chat_messages", [])
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    prompt = st.chat_input(t("assistant_placeholder"))
    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        ans = assistant_answer(prompt, decisions, res, lang)
        st.session_state["chat_messages"].append({"role": "assistant", "content": ans})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            st.write(ans)
