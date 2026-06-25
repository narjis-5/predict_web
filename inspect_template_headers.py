import re
import zipfile
import xml.etree.ElementTree as ET


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def all_text(xml):
    root = ET.fromstring(xml)
    return " ".join(t.text or "" for t in root.findall(".//w:t", NS)).strip()


with zipfile.ZipFile("PFE version finale.docx") as z:
    rel = z.read("word/_rels/document.xml.rels").decode("utf-8", errors="ignore")
    print("REL_HEADERS")
    for m in re.finditer(r'<Relationship[^>]+Id="([^"]+)"[^>]+Type="[^"]+/header"[^>]+Target="([^"]+)"', rel):
        rid, target = m.groups()
        xml = z.read("word/" + target).decode("utf-8", errors="ignore")
        text = all_text(xml)
        print(rid, target, text[:160])

    print("REL_FOOTERS")
    for m in re.finditer(r'<Relationship[^>]+Id="([^"]+)"[^>]+Type="[^"]+/footer"[^>]+Target="([^"]+)"', rel):
        rid, target = m.groups()
        xml = z.read("word/" + target).decode("utf-8", errors="ignore")
        text = all_text(xml)
        print(rid, target, text[:160])
