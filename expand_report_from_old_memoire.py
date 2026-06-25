import html
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "memoire_final_etendu2.docx"
MD = ROOT / "rapport_pfe_source_redaction.md"

START = "<!-- OLD_MEMOIRE_EXPANSION_START -->"
END = "<!-- OLD_MEMOIRE_EXPANSION_END -->"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def docx_text(path):
    """Extract only visible Word text nodes; never copy Word XML markup as text."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    text = " ".join(html.unescape(t.text or "") for t in root.findall(".//w:t", NS))
    return re.sub(r"\s+", " ", text).strip()


def between(text, start, end):
    i = text.find(start)
    j = text.find(end, i + len(start)) if i >= 0 else -1
    if i < 0 or j < 0:
        return ""
    return text[i:j].strip()


def strip_word_xml_leaks(text):
    text = re.sub(r"<w:[^>]+>", " ", text)
    text = re.sub(r"</w:[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def paragraphize(text, sentences_per_paragraph=5):
    text = strip_word_xml_leaks(text)
    text = re.sub(r"\bCHAPITRE\s+\d+\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\bSection\s+\d+(\.\d+)*\s*:\s*", "", text, flags=re.I)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    paragraphs = []
    buf = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        buf.append(sentence)
        if len(buf) >= sentences_per_paragraph:
            paragraphs.append(" ".join(buf))
            buf = []
    if buf:
        paragraphs.append(" ".join(buf))
    return "\n\n".join(paragraphs)


def insert_before_synthesis(md, chapter_heading, addition):
    chapter_heading = chapter_heading[3:] if chapter_heading.startswith("## ") else chapter_heading
    m = re.search(rf"^## {re.escape(chapter_heading)}", md, flags=re.M)
    if not m:
        return md
    start = m.start()
    next_chapter = md.find("\n## Chapitre", start + len(chapter_heading))
    end = next_chapter if next_chapter >= 0 else len(md)
    block = md[start:end]
    synth = re.search(r"\n### Synth\S*se du chapitre", block)
    if not synth:
        return md[:end] + "\n\n" + addition + md[end:]
    insert_at = start + synth.start()
    return md[:insert_at] + "\n\n" + addition + "\n" + md[insert_at:]


def main():
    md = MD.read_text(encoding="utf-8")
    md = re.sub(rf"\n?{re.escape(START)}[\s\S]*?{re.escape(END)}\n?", "\n\n", md)

    old = docx_text(OLD)
    ch1 = between(old, "CHAPITRE 1:", "CHAPITRE 2:")
    ch2 = between(old, "CHAPITRE 2:", "CHAPITRE 3:")
    ch3 = between(old, "CHAPITRE 3:", "BIBLIOGRAPHIE DE LA PARTIE I")

    chapter1_addition = (
        f"{START}\n"
        "### 1.5 Approfondissement théorique intégré depuis le mémoire initial\n\n"
        "Le développement suivant reprend et réorganise le cadre théorique déjà rédigé dans le mémoire initial. "
        "Il renforce le chapitre sans modifier son axe : comprendre pourquoi le dernier kilomètre e-commerce rend "
        "la promesse ETA instable et difficile à contrôler.\n\n"
        + paragraphize(ch1 + " " + ch2, 5)
        + f"\n{END}"
    )

    chapter2_addition = (
        f"{START}\n"
        "### 2.7 Approfondissement théorique intégré depuis le mémoire initial\n\n"
        "Le développement suivant prolonge le cadrage technologique et méthodologique. Il relie la Logistique 4.0, "
        "l'exploitation des traces numériques et les exigences de validation qui justifient le pipeline retenu "
        "dans la partie pratique.\n\n"
        + paragraphize(ch3, 5)
        + f"\n{END}"
    )

    md = insert_before_synthesis(md, "## Chapitre 1 -", chapter1_addition)
    md = insert_before_synthesis(md, "## Chapitre 2 -", chapter2_addition)
    MD.write_text(md, encoding="utf-8")
    print("Expanded", len(md.split()), "words")


if __name__ == "__main__":
    main()
