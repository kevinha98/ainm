"""Quick extraction function tests."""
import sys
sys.path.insert(0, ".")
from agent_v2 import detect_task_type, extract_name, extract_company_name, extract_date

print("=== Name Extraction ===")
tests = [
    ("Opprett en ansatt med navn Ola Nordmann, ola@example.org", ("Ola", "Nordmann")),
    ("Create employee named Maria Hansen, m@t.com", ("Maria", "Hansen")),
    ("Ansett José García-López, jose@test.com", ("José", "García-López")),
    ("Legg til ansatt Åse Ødegård med epost aase@firma.no", ("Åse", "Ødegård")),
]
for text, expected in tests:
    result = extract_name(text)
    ok = "OK" if result == expected else f"FAIL (got {result})"
    print(f"  {ok}: {text[:60]}")

print("\n=== Company Name ===")
for text in [
    "Register a new customer named Test Corp AS with email info@test.no",
    "Opprett kunde Nordlys AS med e-post post@nordlys.no",
    "customer named Acme Ltd with org 987654321",
    "Erstelle Kunde für Müller GmbH, mail@muller.de",
]:
    r = extract_company_name(text)
    print(f"  \"{r}\": {text[:65]}")

print("\n=== Date Parsing ===")
for text in [
    "dato 2026-03-20",
    "15. mars 2026",
    "March 15, 2026",
    "20/03/2026",
    "15 de marzo de 2026",
    "den 5. februar 2026",
    "am 10. Januar 2026",
]:
    r = extract_date(text)
    print(f"  {r}: {text}")

print("\n=== Task Detection ===")
for text in [
    "Opprett en ansatt",
    "Opprett kreditnota for faktura",
    "Slett alle vouchers",
    "Registrer betaling",
    "Opprett reiseregning",
    "Create an invoice for customer",
    "Register a new customer",
    "Godkjenn reiseregning",
    "Send faktura",
    "Update employee information",
    "Opprett ein tilsett Maria Øye",
    "Créer un employé Jean Dupont",
    "Criar cliente Empresa XYZ",
]:
    r = detect_task_type(text)
    print(f"  {r}: {text}")

# Test address extraction
from agent_v2 import extract_address, extract_payment_terms

print("\n=== Address Extraction ===")
for text in [
    "adresse Karl Johans gate 1, 0154 Oslo",
    "address 5th Avenue 100, 10001 New York",
    "Adresse: Storgata 22, 7010 Trondheim",
]:
    r = extract_address(text)
    print(f"  {r}: {text[:50]}")

# Test payment terms
print("\n=== Payment Terms ===")
for text in [
    "betalingsfrist 14 dager",
    "payment terms 30 days",
]:
    r = extract_payment_terms(text)
    print(f"  {r}: {text}")

print("\nDone.")
