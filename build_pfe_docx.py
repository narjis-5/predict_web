import csv
import html
import re
import shutil
import struct
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
MD = ROOT / "rapport_pfe_source_redaction.md"
TEMPLATE = ROOT / "PFE version finale.docx"
OUT = ROOT / "PFE_LOGIPREDICT_version_redigee.docx"
FIG_DIR = ROOT / "report_assets" / "figures"
TABLE_DIR = ROOT / "report_assets" / "tables"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


FIG_CAPTIONS = {
    "01_distribution_durees.png": "Distribution des durees planifiees et reelles.",
    "02_planifie_vs_reel.png": "Relation entre la duree planifiee et la duree reelle.",
    "03_reel_vs_predit.png": "Duree reelle comparee a la duree predite.",
    "04_comparaison_r2.png": "Comparaison du R2 du modele et du baseline metier.",
    "05_importance_permutation.png": "Importance des variables par permutation.",
    "06_matrice_confusion_regle_eta.png": "Matrice de confusion de la regle de risque ETA.",
    "07_distribution_marge_eta.png": "Distribution de la marge ETA predite.",
    "08_groupes_prioritaires.png": "Groupes operationnels prioritaires.",
    "09_ishikawa_retards_eta.png": "Diagramme d'Ishikawa des facteurs de depassement ETA.",
    "21_theorie_chaine_eta.png": "Chaîne de formation et de dégradation de la promesse ETA.",
    "22_theorie_pipeline_ml.png": "Architecture méthodologique du pipeline prédictif.",
    "10_site_accueil_upload.png": "Page d'accueil de LOGI-PREDICT apres import du fichier.",
    "11_site_table_predictions.png": "Apercu des predictions ligne par ligne.",
    "12_site_aide_decision.png": "Onglet d'aide a la decision operationnelle.",
    "13_site_table_decision.png": "Table de decision par axe et transporteur.",
    "14_site_risques_axes_transporteurs.png": "Risques moyens par axe et par transporteur.",
    "15_site_matrice_risque_eta.png": "Matrice risque et tampon ETA.",
    "16_site_distribution_marge_eta.png": "Distribution de l'ecart entre duree predite et ETA actuelle.",
    "17_site_journal_preparation.png": "Journal de preparation du fichier importe.",
    "18_site_modele_interpretation.png": "Validation experte affichee dans l'application.",
    "19_site_importance_validation.png": "Importance des variables dans l'interface de deploiement.",
    "20_site_assistant_decisionnel.png": "Assistant decisionnel local de LOGI-PREDICT.",
}

TABLE_CAPTIONS = {
    "model_metrics.csv": "Performances du modele final selon les protocoles de validation.",
    "risk_rule_metrics.csv": "Evaluation de la regle de risque ETA.",
    "site_summary.csv": "Synthese operationnelle du fichier nigerian importe.",
    "top_decision_groups.csv": "Premiers groupes prioritaires exportes par le site.",
}

COUNTERS = {"figure": 0, "table": 0}


def text_between(text, start_heading, end_heading=None):
    start = text.index(start_heading)
    if end_heading:
        end = text.index(end_heading, start + len(start_heading))
    else:
        end = len(text)
    return text[start:end].strip()


def section_by_heading(text, heading):
    pat = re.compile(rf"^## {re.escape(heading)}\s*$", re.M)
    m = pat.search(text)
    if not m:
        return ""
    n = re.search(r"^## .+$", text[m.end():], re.M)
    end = m.end() + n.start() if n else len(text)
    return text[m.end():end].strip()


def section_heading_prefix(text, heading_prefix):
    pat = re.compile(rf"^## ({re.escape(heading_prefix)}.+)$", re.M)
    m = pat.search(text)
    if not m:
        return ""
    n = re.search(r"^## .+$", text[m.end():], re.M)
    end = m.end() + n.start() if n else len(text)
    return "## " + m.group(1) + "\n" + text[m.end():end].strip()


def clean_inline(s):
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = html.unescape(s)
    return s.strip()


def p(text="", style=None, align=None, bold=False, italic=False, size=None):
    props = []
    if style:
        props.append(f'<w:pStyle w:val="{escape(style)}"/>')
    if align:
        props.append(f'<w:jc w:val="{align}"/>')
    ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
    rprops = []
    if bold:
        rprops.append("<w:b/>")
    if italic:
        rprops.append("<w:i/>")
    if size:
        rprops.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    rpr = f"<w:rPr>{''.join(rprops)}</w:rPr>" if rprops else ""
    return f"<w:p>{ppr}<w:r>{rpr}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"


def page_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def section_break(header_rid=None, footer_rid=None, num_format=None, start=None):
    refs = []
    if header_rid:
        refs.append(f'<w:headerReference w:type="default" r:id="{header_rid}"/>')
    if footer_rid:
        refs.append(f'<w:footerReference w:type="default" r:id="{footer_rid}"/>')
    return (
        "<w:p><w:pPr><w:sectPr>"
        + "".join(refs)
        + '<w:pgSz w:w="12240" w:h="15840"/>'
        + '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
        + (
            f'<w:pgNumType{(" w:fmt=\"" + num_format + "\"") if num_format else ""}{(" w:start=\"" + str(start) + "\"") if start is not None else ""}/>'
            if num_format or start is not None
            else ""
        )
        + '<w:cols w:space="720"/><w:docGrid w:linePitch="360"/>'
        + "</w:sectPr></w:pPr></w:p>"
    )


def field_paragraph(instr, placeholder):
    return (
        '<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve">{escape(instr)}</w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f'<w:r><w:t>{escape(placeholder)}</w:t></w:r>'
        '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
    )


def header_xml(title):
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="{NS['w']}" xmlns:r="{NS['r']}">
  <w:p>
    <w:pPr>
      <w:pBdr><w:bottom w:val="single" w:sz="8" w:space="4" w:color="4F81BD"/></w:pBdr>
      <w:jc w:val="center"/>
    </w:pPr>
    <w:r>
      <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="20"/></w:rPr>
      <w:t>{escape(title)}</w:t>
    </w:r>
  </w:p>
</w:hdr>'''


def footer_xml():
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{NS['w']}" xmlns:r="{NS['r']}">
  <w:p>
    <w:pPr><w:jc w:val="center"/></w:pPr>
    <w:r><w:fldChar w:fldCharType="begin"/></w:r>
    <w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>
    <w:r><w:fldChar w:fldCharType="separate"/></w:r>
    <w:r><w:t>1</w:t></w:r>
    <w:r><w:fldChar w:fldCharType="end"/></w:r>
  </w:p>
</w:ftr>'''


def chapter_title_page(label, title):
    return "\n".join(
        [
            '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
            '<w:p><w:pPr><w:spacing w:before="2400"/></w:pPr></w:p>',
            '<w:tbl><w:tblPr><w:tblW w:w="8500" w:type="dxa"/><w:jc w:val="center"/>'
            '<w:tblBorders><w:top w:val="single" w:sz="24" w:color="ED7D31"/>'
            '<w:left w:val="single" w:sz="24" w:color="5B9BD5"/>'
            '<w:bottom w:val="nil"/><w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders></w:tblPr>'
            '<w:tr><w:tc><w:tcPr><w:tcW w:w="8500" w:type="dxa"/></w:tcPr>'
            f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="360" w:after="240"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="36"/></w:rPr><w:t>{escape(label)}</w:t></w:r></w:p>'
            f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="420"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="30"/></w:rPr><w:t>{escape(title)}</w:t></w:r></w:p>'
            '</w:tc></w:tr></w:tbl>',
            '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
        ]
    )


def strip_first_heading(md_text):
    lines = md_text.splitlines()
    if lines and lines[0].startswith("## "):
        return "\n".join(lines[1:]).lstrip()
    return md_text


def png_size(path):
    with open(path, "rb") as f:
        sig = f.read(24)
    if sig[:8] != b"\x89PNG\r\n\x1a\n":
        return 1200, 700
    return struct.unpack(">II", sig[16:24])


def image_xml(path, rid, docpr_id):
    w_px, h_px = png_size(path)
    max_w = 6.25
    max_h = 4.8
    width_in = max_w
    height_in = width_in * h_px / max(w_px, 1)
    if height_in > max_h:
        height_in = max_h
        width_in = height_in * w_px / max(h_px, 1)
    cx = int(width_in * 914400)
    cy = int(height_in * 914400)
    name = escape(path.name)
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{docpr_id}" name="{name}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr><pic:cNvPr id="0" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>
              <pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
              <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>"""


def table_xml(csv_path, max_rows=14):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.reader(f)):
            if i > max_rows:
                break
            rows.append(row)
    return rows_table_xml(rows)


def rows_table_xml(rows):
    if not rows:
        return ""
    cols = max(len(r) for r in rows)
    out = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/></w:tblPr>']
    for ri, row in enumerate(rows):
        out.append("<w:tr>")
        for ci in range(cols):
            val = clean_inline(row[ci]) if ci < len(row) else ""
            shade = '<w:shd w:fill="D9EAF7"/>' if ri == 0 else ""
            bold = "<w:b/>" if ri == 0 else ""
            out.append(
                f'<w:tc><w:tcPr><w:tcW w:w="2200" w:type="dxa"/>{shade}</w:tcPr>'
                f'<w:p><w:r><w:rPr>{bold}<w:sz w:val="18"/></w:rPr><w:t>{escape(val)}</w:t></w:r></w:p></w:tc>'
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def md_to_ooxml(md_text, rels):
    body = []
    docpr = 10

    in_code = False
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            body.append(p(line, style="Normal", italic=True, size="20"))
            i += 1
            continue
        if line.startswith("|"):
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [clean_inline(c.strip()) for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
                    table_rows.append(cells)
                i += 1
            body.append(rows_table_xml(table_rows))
            continue
        if line.startswith("**Figure à insérer"):
            m = re.search(r"`([^`]+)`", line)
            if m:
                img = ROOT / m.group(1)
                if img.exists():
                    rid = f"rIdImg{len(rels) + 1}"
                    rels[rid] = f"media/{img.name}"
                    body.append(image_xml(img, rid, docpr))
                    docpr += 1
                    COUNTERS["figure"] += 1
                    title = FIG_CAPTIONS.get(img.name, img.name)
                    body.append(p(f"Figure {COUNTERS['figure']}. {title}", style="Caption", align="center", italic=True, size="20"))
            i += 1
            continue
        if line.startswith("**Tableau à insérer"):
            m = re.search(r"`([^`]+)`", line)
            if m:
                tab = ROOT / m.group(1)
                if tab.exists():
                    COUNTERS["table"] += 1
                    title = TABLE_CAPTIONS.get(tab.name, tab.name)
                    body.append(p(f"Tableau {COUNTERS['table']}. {title}", style="Caption", align="center", italic=True, size="20"))
                    body.append(table_xml(tab))
            i += 1
            continue
        if line.startswith("# "):
            body.append(p(clean_inline(line[2:]), style="Titre", align="center", bold=True, size="32"))
        elif line.startswith("## "):
            body.append(p(clean_inline(line[3:]), style="Titre1", bold=True, size="30"))
        elif line.startswith("### "):
            title = clean_inline(line[4:])
            if title.lower() == "introduction":
                body.append(p(title, style="Titre2", bold=True, size="26"))
            else:
                body.append(p(title, style="Titre2", bold=True, size="26"))
        elif line.startswith("- "):
            body.append(p("• " + clean_inline(line[2:]), style="Normal"))
        else:
            body.append(p(clean_inline(line), style="Normal"))
        i += 1
    return "\n".join(body)


def add_cover(parts):
    parts.append(p("Mémoire de fin d'études en vue d'obtention du diplôme de master en ingénierie de la décision", align="center", bold=True, size="28"))
    parts.append(p("", align="center"))
    parts.append(p("Entraînement d'un modèle de Machine Learning pour la prédiction et le recalibrage des délais de livraison e-commerce", align="center", bold=True, size="36"))
    parts.append(p("", align="center"))
    parts.append(p("Présenté par : OURIZ NARJISSE", align="center", size="26"))
    parts.append(p("Encadré par : Pr. BENBERAHIM HOUSSAM", align="center", size="26"))
    parts.append(p("Filière : Master en ingénierie de la décision", align="center", size="24"))
    parts.append(p("Année universitaire : 2026-2027", align="center", size="24"))
    parts.append(page_break())


def add_lists(parts):
    parts.append(p("Table des matières", style="Titre1", bold=True, size="30"))
    parts.append(field_paragraph('TOC \\o "1-3" \\h \\z \\u', "Cliquez avec le bouton droit puis choisissez Mettre à jour les champs dans Word."))
    parts.append(page_break())
    parts.append(p("Liste des figures", style="Titre1", bold=True, size="30"))
    figure_titles = [
        "Diagramme d'Ishikawa des facteurs de dépassement ETA.",
        "Chaîne de formation et de dégradation de la promesse ETA.",
        "Architecture méthodologique du pipeline prédictif.",
        "Distribution des durées planifiées et réelles.",
        "Relation entre la durée planifiée et la durée réelle.",
        "Comparaison du R2 du modèle et du baseline métier.",
        "Durée réelle comparée à la durée prédite.",
        "Importance des variables par permutation.",
        "Matrice de confusion de la règle de risque ETA.",
        "Distribution de la marge ETA prédite.",
        "Groupes opérationnels prioritaires.",
        "Page d'accueil de LOGI-PREDICT après import du fichier.",
        "Aperçu des prédictions ligne par ligne.",
        "Onglet d'aide à la décision opérationnelle.",
        "Table de décision par axe et transporteur.",
        "Risques moyens par axe et par transporteur.",
        "Matrice risque et tampon ETA.",
        "Distribution de l'écart entre durée prédite et ETA actuelle.",
        "Journal de préparation du fichier importé.",
        "Validation experte affichée dans l'application.",
        "Importance des variables dans l'interface de déploiement.",
        "Assistant décisionnel local de LOGI-PREDICT.",
    ]
    for idx, title in enumerate(figure_titles, 1):
        parts.append(p(f"Figure {idx}. {title}"))
    parts.append(page_break())
    parts.append(p("Liste des tableaux", style="Titre1", bold=True, size="30"))
    table_titles = [
        "Performances du modèle final selon les protocoles de validation.",
        "Évaluation de la règle de risque ETA.",
        "Synthèse opérationnelle du fichier nigérian importé.",
        "Premiers groupes prioritaires exportés par le site.",
    ]
    for idx, title in enumerate(table_titles, 1):
        parts.append(p(f"Tableau {idx}. {title}"))
    parts.append(page_break())


def main():
    COUNTERS["figure"] = 0
    COUNTERS["table"] = 0
    text = MD.read_text(encoding="utf-8")
    rels = {}
    extra_parts = {}
    footer_rid = "rIdLogiFooter"
    headers = {
        "intro": ("rIdLogiHeaderIntro", "header_logi_intro.xml", "Introduction Générale"),
        "ch1": ("rIdLogiHeaderCh1", "header_logi_ch1.xml", "Chapitre 1 - Logistique e-commerce et délais"),
        "ch2": ("rIdLogiHeaderCh2", "header_logi_ch2.xml", "Chapitre 2 - Machine Learning prédictif"),
        "ch3": ("rIdLogiHeaderCh3", "header_logi_ch3.xml", "Chapitre 3 - Méthodologie expérimentale"),
        "ch4": ("rIdLogiHeaderCh4", "header_logi_ch4.xml", "Chapitre 4 - Déploiement Streamlit"),
        "conclusion": ("rIdLogiHeaderConclusion", "header_logi_conclusion.xml", "Conclusion Générale"),
        "bibliographie": ("rIdLogiHeaderBiblio", "header_logi_biblio.xml", "Bibliographie et annexes"),
    }
    for _, target, title in headers.values():
        extra_parts[f"word/{target}"] = header_xml(title).encode("utf-8")
    extra_parts["word/footer_logi.xml"] = footer_xml().encode("utf-8")
    parts = []
    add_cover(parts)
    parts.append(p("Dédicaces", style="Titre1", bold=True, size="30"))
    parts.append(p("À ma famille, pour son soutien constant, sa patience et sa confiance durant tout le parcours de ce travail."))
    parts.append(page_break())
    parts.append(p("Remerciements", style="Titre1", bold=True, size="30"))
    parts.append(p("Je remercie mon encadrant, Pr. BENBERAHIM HOUSSAM, pour son accompagnement, ses orientations et ses remarques méthodologiques. Je remercie également toutes les personnes qui ont contribué, directement ou indirectement, à l'aboutissement de ce projet."))
    parts.append(page_break())
    parts.append(p("Résumé", style="Titre1", bold=True, size="30"))
    parts.append(md_to_ooxml(section_by_heading(text, "Résumé"), rels))
    parts.append(page_break())
    parts.append(p("Abstract", style="Titre1", bold=True, size="30"))
    parts.append(md_to_ooxml(section_by_heading(text, "Abstract"), rels))
    parts.append(page_break())
    parts.append(p("Liste des abréviations", style="Titre1", bold=True, size="30"))
    parts.append(md_to_ooxml(section_by_heading(text, "Liste des abréviations"), rels))
    parts.append(page_break())
    add_lists(parts)
    parts.append(section_break(None, footer_rid, num_format="lowerRoman", start=1))
    intro = text_between(text, "## Début de l'introduction générale", "## Références vérifiées à utiliser").replace("## Début de l'introduction générale", "## Introduction générale")
    parts.append(chapter_title_page("Introduction Générale", "Problématique, objectif et démarche du projet"))
    parts.append(md_to_ooxml(strip_first_heading(intro), rels))
    parts.append(section_break(headers["intro"][0], footer_rid, num_format="decimal", start=1))
    ch1 = section_heading_prefix(text, "Chapitre 1 -")
    ch2 = section_heading_prefix(text, "Chapitre 2 -")
    ch3 = section_heading_prefix(text, "Chapitre 3 -")
    ch4 = section_heading_prefix(text, "Chapitre 4 -")
    chapter_specs = [
        ("ch1", "Chapitre 1", "Logistique e-commerce, dernier kilomètre et problématique des délais", ch1),
        ("ch2", "Chapitre 2", "Machine Learning appliqué à la prédiction logistique", ch2),
        ("ch3", "Chapitre 3", "Méthodologie expérimentale, entraînement et évaluation du modèle", ch3),
        ("ch4", "Chapitre 4", "Déploiement Streamlit et aide à la décision opérationnelle", ch4),
    ]
    for key, label, title, chapter in chapter_specs:
        parts.append(chapter_title_page(label, title))
        parts.append(md_to_ooxml(strip_first_heading(chapter), rels))
        parts.append(section_break(headers[key][0], footer_rid))
    parts.append(chapter_title_page("Conclusion Générale", "Apports, limites et perspectives"))
    parts.append(md_to_ooxml(section_by_heading(text, "Conclusion générale"), rels))
    parts.append(section_break(headers["conclusion"][0], footer_rid))
    parts.append(chapter_title_page("Bibliographie", "Références scientifiques et sources méthodologiques"))
    parts.append(md_to_ooxml("## Bibliographie\n\n" + section_by_heading(text, "Bibliographie finale avec liens de téléchargement ou de consultation"), rels))
    parts.append(section_break(headers["bibliographie"][0], footer_rid))
    parts.append(md_to_ooxml("## Annexes\n\n" + section_by_heading(text, "Annexes à intégrer"), rels))
    parts.append(section_break(headers["bibliographie"][0], footer_rid))

    with zipfile.ZipFile(TEMPLATE, "r") as zin:
        original_doc = zin.read("word/document.xml").decode("utf-8")
        sectprs = re.findall(r"<w:sectPr\b[\s\S]*?</w:sectPr>", original_doc)
        sectpr = sectprs[-1] if sectprs else "<w:sectPr/>"
        body_xml = "\n".join(parts) + "\n" + sectpr
        document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS['w']}" xmlns:r="{NS['r']}" xmlns:wp="{NS['wp']}" xmlns:a="{NS['a']}" xmlns:pic="{NS['pic']}">
<w:body>{body_xml}</w:body></w:document>'''

        rel_xml = zin.read("word/_rels/document.xml.rels").decode("utf-8")
        rel_root = ET.fromstring(rel_xml)
        rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
        for rid, target, _ in headers.values():
            ET.SubElement(
                rel_root,
                f"{{{rel_ns}}}Relationship",
                {
                    "Id": rid,
                    "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header",
                    "Target": target,
                },
            )
        ET.SubElement(
            rel_root,
            f"{{{rel_ns}}}Relationship",
            {
                "Id": footer_rid,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer",
                "Target": "footer_logi.xml",
            },
        )
        for rid, target in rels.items():
            ET.SubElement(
                rel_root,
                f"{{{rel_ns}}}Relationship",
                {
                    "Id": rid,
                    "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                    "Target": target,
                },
            )
        new_rel_xml = ET.tostring(rel_root, encoding="utf-8", xml_declaration=True)

        content_types = zin.read("[Content_Types].xml").decode("utf-8")
        if 'Extension="png"' not in content_types:
            content_types = content_types.replace(
                "</Types>",
                '<Default Extension="png" ContentType="image/png"/></Types>',
            )
        if "header_logi_intro.xml" not in content_types:
            overrides = []
            for _, target, _ in headers.values():
                overrides.append(
                    f'<Override PartName="/word/{target}" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>'
                )
            overrides.append(
                '<Override PartName="/word/footer_logi.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
            )
            content_types = content_types.replace("</Types>", "".join(overrides) + "</Types>")

        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, document_xml.encode("utf-8"))
                elif item.filename == "word/_rels/document.xml.rels":
                    zout.writestr(item, new_rel_xml)
                elif item.filename == "[Content_Types].xml":
                    zout.writestr(item, content_types.encode("utf-8"))
                else:
                    zout.writestr(item, zin.read(item.filename))
            for name, payload in extra_parts.items():
                zout.writestr(name, payload)
            for rid, target in rels.items():
                name = Path(target).name
                img = FIG_DIR / name
                if img.exists():
                    zout.write(img, f"word/{target}")

    print(OUT)


if __name__ == "__main__":
    main()
