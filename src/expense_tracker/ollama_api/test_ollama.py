from pathlib import Path

from expense_tracker.ollama_api.assistant import OllamaAssistant

project_root = Path(__file__).resolve().parents[3]
image_path = project_root / "output/abstract_layout_images/page_001_abstract.png"
image_original_path = project_root / "data/debug/image3_normalized.jpg"
ocr_json_path = project_root / "data/debug/image3_ocr.json"

inference_assistant = OllamaAssistant()

abstract_layout_response = inference_assistant.infer(
    prompt="""
Analysiere ausschließlich dieses abstrakte OCR-Layout eines Kassenbons.
Extrahiere die erkannten Artikel, Mengen und Preise. Beschreibe Unsicherheiten
und antworte als valides JSON.
""",
    images=[image_path],
)

original_image_response = inference_assistant.infer(
    prompt="""
Analysiere ausschließlich dieses Originalbild eines Kassenbons.
Extrahiere die Artikel, Mengen und Preise. Beschreibe Unsicherheiten
und antworte als valides JSON.
""",
    images=[image_original_path],
)

ocr_json_response = inference_assistant.infer(
    prompt="""
Analysiere ausschließlich dieses OCR-JSON eines Kassenbons.
Extrahiere die Artikel, Mengen und Preise. Beschreibe Unsicherheiten
und antworte als valides JSON.
""",
    files=[ocr_json_path],
)

comparison_assistant = OllamaAssistant()
comparison_response = comparison_assistant.chat(
    prompt=f"""
Du erhältst drei unabhängige Voranalysen desselben Kassenbons.
Sie wurden in getrennten Inferenzaufrufen erzeugt und dürfen nicht als
eine gemeinsame Gesprächshistorie interpretiert werden.

--- Antwort 1: abstraktes OCR-Layout ---
{abstract_layout_response}
--- Ende Antwort 1 ---

--- Antwort 2: Originalbild ---
{original_image_response}
--- Ende Antwort 2 ---

--- Antwort 3: OCR-JSON ---
{ocr_json_response}
--- Ende Antwort 3 ---

Vergleiche die drei Antworten. Erkläre:
1. Welche Artikel, Mengen und Preise übereinstimmen.
2. Wo Unterschiede oder Widersprüche bestehen.
3. Welche Quelle für die finale Extraktion am zuverlässigsten ist.
4. Wie die Quellen gemeinsam validiert werden sollten.

Gib danach eine konsolidierte Extraktion des Kassenbons aus.
""",
)

print("\n--- Abstraktes OCR-Layout ---")
print(abstract_layout_response)
print("\n--- Originalbild ---")
print(original_image_response)
print("\n--- OCR-JSON ---")
print(ocr_json_response)
print("\n--- Vergleich und konsolidierte Extraktion ---")
print(comparison_response)