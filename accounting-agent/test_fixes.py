"""Test all extraction functions and task detection."""
import agent_v2

# Test task detection
tests = [
    ('Opprett en ansatt med navn Test Person, test@example.no', 'create_employee'),
    ('Erstellen Sie einen Mitarbeiter Max Müller, max@test.de', 'create_employee'),
    ('Erstellen Sie das Produkt Orangensaft mit der Produktnummer 1256', 'create_product'),
    ('Create an invoice for customer Testadr AS for 5000 NOK', 'create_invoice'),
    ('Erstellen Sie einen Kunden Müller GmbH', 'create_customer'),
    ('Opprett ein avdeling med namn Salg', 'create_department'),
    ('Créez un employé Pierre Dupont, pierre@test.fr', 'create_employee'),
    ('Cree un cliente XYZ SL con email info@xyz.es', 'create_customer'),
    ('Erstellen Sie eine Rechnung für Kunden ABC AG', 'create_invoice'),
    ('Send fakturaen til kunden', 'send_invoice'),
    ('Godkjenn reiseregning', 'approve_travel_expense'),
    ('Slett alle bilag', 'delete_voucher'),
    ('Opprett kreditnota for faktura 123', 'create_credit_note'),
    ('Erstellen Sie eine Gutschrift', 'create_credit_note'),
    ('Löschen Sie die Reisekostenabrechnung', 'delete_travel_expense'),
    ('Erstellen Sie einen Lieferanten Müller GmbH', 'create_supplier'),
    ('Crie um funcionário Maria Silva, maria@test.pt', 'create_employee'),
    ('Opprett ein leverandør med namn ABC AS', 'create_supplier'),
    ('Erstellen Sie ein Projekt TestProjekt', 'create_project'),
    ('Registrer betaling for faktura 456', 'register_payment'),
    ('Senden Sie die Rechnung', 'send_invoice'),
    ('Genehmigen Sie die Reisekostenabrechnung', 'approve_travel_expense'),
    ('Créez un contact Jean Martin', 'create_contact'),
    ('Oppdater ansatt Test Person', 'update_employee'),
    ('Aktualisieren Sie den Kunden Müller GmbH', 'update_customer'),
]

passed = 0
for prompt, expected in tests:
    result = agent_v2.detect_task_type(prompt)
    ok = result == expected
    passed += ok
    if not ok:
        print(f'  FAIL: "{prompt[:60]}" → {result} (expected {expected})')

print(f'{passed}/{len(tests)} task detection tests passed')
print()

# Test name extraction
name_tests = [
    ('Erstellen Sie einen Mitarbeiter Max Müller, max@test.de', ('Max', 'Müller')),
    ('Opprett ansatt med namn Ola Nordmann', ('Ola', 'Nordmann')),
    ('Créez un employé Pierre Dupont', ('Pierre', 'Dupont')),
    ('Create employee John Smith, john@test.com', ('John', 'Smith')),
    ('Crie um funcionário Maria Silva', ('Maria', 'Silva')),
]
for prompt, expected in name_tests:
    result = agent_v2.extract_name(prompt)
    ok = result == expected
    if not ok:
        print(f'  FAIL name: "{prompt[:60]}" → {result} (expected {expected})')
    else:
        print(f'  OK name: {result}')

print()

# Test company name extraction
comp_tests = [
    ('Erstellen Sie einen Kunden Müller GmbH', 'Müller GmbH'),
    ('Opprett kunde Testfirma AS med e-post', 'Testfirma AS'),
    ('Create customer ABC Ltd with email', 'ABC Ltd'),
    ('customer "Nordic Solutions AS" with org', 'Nordic Solutions AS'),
    ('Créez un client Dupont SARL avec email', 'Dupont SARL'),
]
for prompt, expected in comp_tests:
    result = agent_v2.extract_company_name(prompt)
    ok = result == expected
    if not ok:
        print(f'  FAIL company: "{prompt[:60]}" → {result} (expected {expected})')
    else:
        print(f'  OK company: {result}')

print()

# Test product extraction
prod_tests = [
    ('Erstellen Sie das Produkt "Orangensaft" mit der Produktnummer 1256. Der Preis beträgt 17450 NOK', 
     {'name': 'Orangensaft', 'number': '1256', 'price': 17450.0}),
    ('Opprett produkt "Kaffe" med pris 50 kr', 
     {'name': 'Kaffe', 'price': 50.0}),
]
for prompt, expected in prod_tests:
    result = agent_v2.extract_product_info(prompt)
    for k, v in expected.items():
        if result.get(k) != v:
            print(f'  FAIL product {k}: got {result.get(k)}, expected {v} (from: "{prompt[:50]}")')
        else:
            print(f'  OK product {k}: {result.get(k)}')
