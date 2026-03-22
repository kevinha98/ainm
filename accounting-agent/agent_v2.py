"""
AI Accounting Agent for Tripletex competition.
Hybrid: regex-based task detection + entity extraction, with optional LLM fallback.
"""
import base64
import datetime
import json
import logging
import os
import re
from typing import Any

from tripletex import TripletexClient

logger = logging.getLogger(__name__)

# ─── Multilingual keyword maps for task detection ───────────────────────────
TASK_KEYWORDS = {
    "create_employee": [
        "opprett ansatt", "opprett en ansatt", "ny ansatt", "registrer ansatt",
        "ansatt med namn", "ansatt med namn", "ansatt med navn",
        "opprett tilsett", "opprett ein tilsett", "ny tilsett",
        "create employee", "create an employee", "new employee", "register employee",
        "mitarbeiter erstellen", "neuen mitarbeiter", "erstellen sie einen mitarbeiter",
        "erstellen sie eine mitarbeiterin", "legen sie einen mitarbeiter",
        "créer employé", "créer un employé", "nouvel employé", "créez un employé",
        "crear empleado", "nuevo empleado", "cree un empleado", "crear un empleado",
        "criar funcionário", "novo funcionário", "crie um funcionário",
    ],
    "create_customer": [
        "opprett kunde", "opprett en kunde", "ny kunde", "registrer kunde",
        "opprett ein kunde", "registrer en ny kunde",
        "create customer", "create a customer", "new customer", "register customer",
        "kunde erstellen", "neuen kunden", "erstellen sie einen kunden",
        "legen sie einen kunden", "erstellen sie den kunden",
        "créer client", "créer un client", "nouveau client", "créez un client",
        "crear cliente", "nuevo cliente", "cree un cliente",
        "criar cliente", "novo cliente", "crie um cliente",
    ],
    "create_product": [
        "opprett produkt", "nytt produkt", "registrer produkt", "opprett eit produkt",
        "create product", "new product", "register product", "create a product",
        "create the product",
        "produkt erstellen", "neues produkt", "erstellen sie das produkt",
        "erstellen sie ein produkt", "legen sie ein produkt",
        "créer produit", "nouveau produit", "créez un produit", "créer le produit",
        "crear producto", "nuevo producto", "cree un producto", "crear el producto",
        "criar produto", "novo produto", "crie um produto", "criar o produto",
    ],
    "create_invoice": [
        "opprett faktura", "lag faktura", "ny faktura", "fakturer",
        "opprett ein faktura",
        "create invoice", "create an invoice", "new invoice", "generate invoice", "issue invoice",
        "rechnung erstellen", "neue rechnung", "erstellen sie eine rechnung",
        "créer facture", "nouvelle facture", "créez une facture",
        "crear factura", "nueva factura", "cree una factura",
        "criar fatura", "nova fatura", "crie uma fatura",
    ],
    "send_invoice": [
        "send faktura", "send fakturaen", "send ut faktura",
        "send invoice", "send the invoice",
        "rechnung senden", "rechnung versenden", "senden sie die rechnung",
        "envoyer facture", "envoyer la facture", "envoyez la facture",
        "enviar factura", "envíe la factura",
        "enviar fatura", "envie a fatura",
    ],
    "approve_travel_expense": [
        "godkjenn reiseregning", "godkjenn reiserekning",
        "approve travel expense", "approve travel report", "approve the travel",
        "reisekosten genehmigen", "genehmigen sie die reisekosten",
        "genehmigen reisekosten", "reisekostenabrechnung genehmigen",
        "approuver note de frais", "approuvez la note de frais",
        "aprobar gastos de viaje", "apruebe los gastos",
        "aprovar despesa de viagem", "aprove a despesa",
    ],
    "create_project": [
        "opprett prosjekt", "lag prosjekt", "nytt prosjekt", "opprett eit prosjekt",
        "create project", "new project", "register project", "create a project",
        "projekt erstellen", "neues projekt", "erstellen sie ein projekt",
        "créer projet", "nouveau projet", "créez un projet",
        "crear proyecto", "nuevo proyecto", "cree un proyecto",
        "criar projeto", "novo projeto", "crie um projeto",
    ],
    "create_department": [
        "opprett avdeling", "ny avdeling", "registrer avdeling",
        "opprett ei avdeling", "opprett ein avdeling",
        "create department", "new department", "create a department",
        "abteilung erstellen", "neue abteilung", "erstellen sie eine abteilung",
        "créer département", "créer service", "créer un département",
        "crear departamento", "nuevo departamento", "cree un departamento",
        "criar departamento", "novo departamento", "crie um departamento",
    ],
    "register_payment": [
        "registrer betaling", "registrer innbetaling", "betal faktura", "innbetaling",
        "register payment", "record payment", "pay invoice",
        "zahlung registrieren", "zahlung erfassen", "registrieren sie eine zahlung",
        "enregistrer paiement", "enregistrez un paiement",
        "registrar pago", "registre un pago",
        "registrar pagamento", "registre um pagamento",
    ],
    "create_supplier_invoice": [
        # Norwegian
        "registrer leverandørfaktura", "leverandørfaktura", "inngående faktura",
        "registrer innkjøpsfaktura", "mottatt faktura fra",
        # Nynorsk
        "registrer leverandørfaktura", "innkjøpsfaktura",
        # English
        "supplier invoice", "register supplier invoice", "purchase invoice",
        "incoming invoice", "vendor invoice", "received invoice from",
        "register the invoice", "register the supplier invoice",
        # German
        "lieferantenrechnung", "eingangsrechnung", "einkaufsrechnung",
        "registrieren sie die rechnung",
        # French
        "facture fournisseur", "facture d'achat", "enregistrez la facture",
        "facture entrante", "reçu la facture", "recebemos a fatura",
        # Spanish
        "factura de proveedor", "factura de compra", "registre la factura",
        # Portuguese
        "fatura do fornecedor", "fatura de compra", "recebemos a fatura",
        "registe a fatura",
    ],
    "create_credit_note": [
        "kreditnota", "krediter faktura", "opprett kreditnota", "lag kreditnota",
        "credit note", "create credit note", "issue credit note", "create a credit note",
        "gutschrift erstellen", "gutschrift", "erstellen sie eine gutschrift",
        "note de crédit", "avoir", "créer une note de crédit",
        "nota de crédito", "crear nota de crédito", "cree una nota de crédito",
        "nota de crédito", "criar nota de crédito",
    ],
    "delete_travel_expense": [
        "slett reiseregning", "fjern reiseregning", "slett reise",
        "slett reiserekning", "slett alle reiseregning",
        "delete travel expense", "remove travel expense", "delete the travel",
        "reisekosten löschen", "reisekostenabrechnung löschen", "löschen sie die reisekosten",
        "supprimer note de frais", "supprimez la note de frais",
        "eliminar gastos de viaje", "elimine los gastos",
        "excluir despesa de viagem", "exclua a despesa",
    ],
    "create_travel_expense": [
        "opprett reiseregning", "registrer reiseregning", "ny reiseregning",
        "opprett reiserekning",
        "create travel expense", "register travel expense", "new travel expense",
        "create a travel expense",
        "reisekosten erstellen", "reisekostenabrechnung erstellen",
        "erstellen sie eine reisekostenabrechnung",
        "créer note de frais", "créez une note de frais",
        "crear gastos de viaje", "cree gastos de viaje",
        "criar despesa de viagem", "crie uma despesa de viagem",
    ],
    "create_supplier": [
        "opprett leverandør", "ny leverandør", "registrer leverandør",
        "opprett ein leverandør",
        "create supplier", "new supplier", "register supplier", "create vendor",
        "create a supplier",
        "lieferant erstellen", "neuen lieferanten", "erstellen sie einen lieferanten",
        "créer fournisseur", "nouveau fournisseur", "créez un fournisseur",
        "crear proveedor", "nuevo proveedor", "cree un proveedor",
        "criar fornecedor", "novo fornecedor", "crie um fornecedor",
    ],
    "update_employee": [
        "oppdater ansatt", "endre ansatt", "rediger ansatt",
        "oppdater tilsett", "endre tilsett",
        "update employee", "edit employee", "modify employee", "change employee",
        "mitarbeiter aktualisieren", "aktualisieren sie den mitarbeiter",
        "mitarbeiter ändern", "ändern sie den mitarbeiter",
        "modifier employé", "modifiez l'employé", "mettre à jour l'employé",
        "actualizar empleado", "actualice el empleado", "modificar empleado",
        "atualizar funcionário", "atualize o funcionário",
    ],
    "update_customer": [
        "oppdater kunde", "endre kunde", "rediger kunde",
        "update customer", "edit customer", "modify customer", "change customer",
        "kunde aktualisieren", "aktualisieren sie den kunden",
        "kunde ändern", "ändern sie den kunden",
        "modifier client", "modifiez le client", "mettre à jour le client",
        "actualizar cliente", "actualice el cliente",
        "atualizar cliente", "atualize o cliente",
    ],
    "create_contact": [
        "opprett kontakt", "ny kontaktperson", "legg til kontakt", "kontaktperson",
        "opprett ein kontaktperson",
        "create contact", "add contact", "new contact", "create a contact",
        "kontakt erstellen", "kontaktperson erstellen", "erstellen sie einen kontakt",
        "ansprechpartner erstellen",
        "créer contact", "nouveau contact", "créez un contact",
        "crear contacto", "nuevo contacto", "cree un contacto",
        "criar contato", "novo contato", "crie um contato",
    ],
    "delete_voucher": [
        "slett voucher", "slett bilag", "fjern voucher", "fjern bilag",
        "slett alle voucher", "slett alle bilag",
        "delete voucher", "remove voucher", "delete all vouchers",
        "beleg löschen", "voucher löschen", "löschen sie den beleg",
        "löschen sie alle belege",
        "supprimer pièce", "supprimer écriture", "supprimez les pièces",
        "eliminar comprobante", "borrar comprobante", "elimine los comprobantes",
        "excluir voucher", "excluir comprovante", "exclua os comprovantes",
    ],
    "create_timesheet": [
        "registrer timer", "registrer arbeidstimer", "timeføring", "timeliste", "registrer time",
        "registrer timar", "timeregistrering", "loggfør timer",
        "register hours", "register timesheet", "log hours", "record hours", "create timesheet",
        "stunden erfassen", "stunden registrieren", "arbeitszeit erfassen", "zeiterfassung",
        "erstellen sie eine zeiterfassung", "erfassen sie stunden",
        "enregistrer heures", "registrer des heures", "feuille de temps",
        "registrar horas", "registre horas", "hoja de horas",
        "registrar horas", "registre horas", "folha de ponto",
    ],
    "create_voucher": [
        "opprett bilag", "registrer bilag", "nytt bilag", "bokfør", "bokføring",
        "opprett eit bilag", "lag bilag", "journalpost",
        "create voucher", "create journal entry", "record journal entry", "post journal",
        "beleg erstellen", "buchung erstellen", "erstellen sie einen beleg",
        "erstellen sie eine buchung", "journalbuchung",
        "créer écriture", "pièce comptable", "créez une écriture",
        "comptabilisez une pièce", "comptabilisez", "comptabiliser",
        "crear asiento", "comprobante contable", "cree un asiento",
        "criar lançamento", "comprovante contábil", "crie um lançamento",
    ],
    "enable_module": [
        "aktiver modul", "slå på modul", "aktiver avdeling", "aktiver prosjekt",
        "enable module", "activate module", "enable department", "enable project",
        "modul aktivieren", "aktivieren sie", "abteilung aktivieren",
        "activer module", "activez le module",
        "activar módulo", "active el módulo",
        "ativar módulo", "ative o módulo",
    ],
    "deliver_travel_expense": [
        "lever reiseregning", "lever inn reiseregning", "send inn reiseregning",
        "lever reiserekning",
        "deliver travel expense", "submit travel expense", "submit travel report",
        "reisekosten einreichen", "reichen sie die reisekosten ein",
        "soumettre note de frais", "soumettez la note de frais",
        "enviar gastos de viaje", "envíe los gastos",
        "enviar despesa de viagem", "envie a despesa",
    ],
    "create_dimension": [
        # Norwegian
        "opprett dimensjon", "ny dimensjon", "egendefinert dimensjon",
        "opprett ein dimensjon", "lag dimensjon", "tilpasset dimensjon",
        # English
        "create dimension", "custom dimension", "create a dimension",
        "create custom dimension", "new dimension",
        # German
        "dimension erstellen", "erstellen sie eine dimension",
        "benutzerdefinierte dimension", "neue dimension",
        # French
        "dimension comptable personnalisée", "dimension comptable",
        "dimension personnalisée", "créer dimension", "créez une dimension",
        # Spanish
        "crear dimensión", "dimensión personalizada",
        "cree una dimensión", "dimensión contable",
        # Portuguese
        "criar dimensão", "dimensão personalizada",
        "crie uma dimensão", "dimensão contábil",
    ],
}


# ─── Entity extraction ──────────────────────────────────────────────────────

def extract_email(text: str) -> str | None:
    m = re.search(r'[\w.+-]+@[\w-]+\.[\w]+(?:\.[\w]+)*', text)
    return m.group(0).rstrip('.') if m else None


def extract_phone(text: str) -> str | None:
    m = re.search(r'(?:\+\d{1,3}\s?)?\d[\d\s\-]{6,14}\d', text)
    return re.sub(r'[\s\-]', '', m.group(0)) if m else None


def extract_name(text: str) -> tuple[str, str] | None:
    # Unicode letter class for names (covers æøå, accented chars, etc.)
    CAP = r'[A-ZÆØÅÄÖÜ]'
    LOW = r'[a-zæøåäöüéèêëàâáãçñíìîïóòôõúùûü]'
    NAME = f'{CAP}{LOW}+(?:-{CAP}{LOW}+)?'  # supports hyphenated names like Anne-Marie
    patterns = [
        # "navn Ola Nordmann" / "med navn Ola Nordmann"
        rf'(?:med\s+)?(?:namn|navn[n]?)\s+({NAME})\s+({NAME})',
        # "named/llamado/nommé/namens/chamado FirstName LastName"
        rf'(?:named?|name[d]?|llamado|nommé|namens|chamado|heiter|hei[ßt]t|heißt|heter)\s+({NAME})\s+({NAME})',
        # "ansatt/employee/Mitarbeiter FirstName LastName"
        rf'(?:ansatt|employee|empleado|employé|mitarbeiter(?:in)?|funcionário|tilsett|kontakt|contact|ansprechpartner)\s+({NAME})\s+({NAME})',
        # "opprett en ansatt FirstName LastName"
        rf'(?:opprett|create|lag|registrer|créer|créez|crear|cree|criar|crie|erstellen)\s+(?:en\s+|ein(?:e[n]?)?\s+|une?\s+|uma?\s+|sie\s+)?(?:ny\s+|new\s+|neuen?\s+|nouvel(?:le)?\s+|nuevo\s+|novo\s+)?(?:ansatt|tilsett|employee|mitarbeiter(?:in)?|employé|empleado|funcionário|kontakt|contact|ansprechpartner)\s+(?:med\s+(?:namn|navn)\s+)?({NAME})\s+({NAME})',
        # Two capitalized words before/near an email
        rf'({NAME})\s+({NAME})(?=[,\s]*[\w.+-]+@)',
        # "für Mitarbeiter FirstName LastName" (German)
        rf'(?:für|fur)\s+(?:den\s+)?(?:Mitarbeiter(?:in)?|Angestellte[n]?)\s+({NAME})\s+({NAME})',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            first = m.group(1)
            last = m.group(2)
            # Capitalize properly
            first = first[0].upper() + first[1:]
            last = last[0].upper() + last[1:]
            return first, last
    return None


def extract_company_name(text: str) -> str | None:
    SUFFIX = r'(?:AS|ASA|ANS|DA|NUF|SA|AB|GmbH|AG|e\.V\.|Ltd|LLC|Inc|SL|SRL|SARL|Oy|S\.A\.|S\.L\.)'
    WORD = r'[A-ZÆØÅÄÖÜa-zæøåäöü][\wæøåäöü&.\-]*'
    patterns = [
        # Quoted names (highest priority — catches exact names)
        r'["\u201c\u201d«»]([^"\u201c\u201d«»]+)["\u201c\u201d«»]',
        # customer/client/supplier keyword + name (strips task verbs)  
        rf'(?:kunde[n]?|customer|client|leverandør|supplier|fournisseur|proveedor|fornecedor|Kunde[n]?|Lieferant(?:en)?|Firma|cliente)\s+(?:named?\s+|called\s+|kalt\s+|heter\s+|med\s+(?:namn\s+|navn\s+)?)?({WORD}(?:\s+{WORD}){{0,4}}?)(?=\s*(?:\.|,|med\b|with\b|mit\b|avec\b|con\b|com\b|e-?post|email|telefon|phone|org|for\b|adress|para\b|pour\b|\d{{4}}|$))',
        # "named X", "called X", etc before a delimiter
        r'(?:named?|called|kalt|heter|med\s+namn|med\s+navn|namens|chamado|llamado|nommé)\s+([A-ZÆØÅÄÖÜ][\w\s&.\-]+?)(?=\s*(?:\.|,|med\b|with\b|e-?post|email|telefon|phone|org\.?\s*n|$))',
        # German: "für X"
        rf'(?:für|fur|del? cliente|du client|do cliente)\s+({WORD}(?:\s+{WORD}){{0,3}}?)(?=\s*(?:\.|,|mit\b|with\b|med\b|e-?post|email|telefon|phone|org|adress|$))',
        # Company suffix: 1-3 words directly before suffix (restrictive)
        rf'(?:^|[\s,;(])({WORD}(?:\s+{WORD}){{0,2}}?\s+{SUFFIX})(?=[\s,;).]|$)',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            name = m.group(1).strip().rstrip('.,')
            # Strip any leading common/task words
            skip_words = {'opprett', 'create', 'new', 'ny', 'erstellen', 'créer', 'crear', 'criar',
                          'sie', 'einen', 'eine', 'ein', 'den', 'die', 'das', 'det', 'en', 'et',
                          'kunden', 'kunde', 'customer', 'client', 'for', 'med', 'with'}
            words = name.split()
            while words and words[0].lower() in skip_words:
                words.pop(0)
            if words:
                name = ' '.join(words)
            else:
                continue
            if len(name) < 2:
                continue
            return name
    return None


MONTH_MAP = {
    # Norwegian
    'januar': '01', 'februar': '02', 'mars': '03', 'april': '04', 'mai': '05', 'juni': '06',
    'juli': '07', 'august': '08', 'september': '09', 'oktober': '10', 'november': '11', 'desember': '12',
    # English
    'january': '01', 'february': '02', 'march': '03', 'may': '05', 'june': '06',
    'july': '07', 'october': '10', 'december': '12',
    # German
    'januar': '01', 'märz': '03', 'mai': '05', 'juni': '06', 'juli': '07',
    'oktober': '10', 'dezember': '12',
    # French
    'janvier': '01', 'février': '02', 'avril': '04', 'juin': '06',
    'juillet': '07', 'août': '08', 'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12',
    # Spanish
    'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 'mayo': '05', 'junio': '06',
    'julio': '07', 'agosto': '08', 'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12',
    # Portuguese
    'janeiro': '01', 'fevereiro': '02', 'março': '03', 'maio': '05', 'junho': '06',
    'julho': '07', 'setembro': '09', 'outubro': '10', 'novembro': '11', 'dezembro': '12',
}

def extract_date(text: str) -> str | None:
    # ISO format: 2026-03-20
    m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if m:
        return m.group(1)
    # DD.MM.YYYY or DD/MM/YYYY
    m = re.search(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', text)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    # Written month: "15. mars 2026", "March 15, 2026", "15 de marzo de 2026"
    m = re.search(r'(\d{1,2})\.?\s+(?:de\s+)?([a-zæøåäöüéèê]+)\s+(?:de\s+)?(\d{4})', text, re.IGNORECASE)
    if m:
        month_name = m.group(2).lower()
        month_num = MONTH_MAP.get(month_name)
        if month_num:
            return f"{m.group(3)}-{month_num}-{m.group(1).zfill(2)}"
    # "March 15, 2026" (English month-first)
    m = re.search(r'([a-zæøåäöüéèê]+)\s+(\d{1,2}),?\s+(\d{4})', text, re.IGNORECASE)
    if m:
        month_name = m.group(1).lower()
        month_num = MONTH_MAP.get(month_name)
        if month_num:
            return f"{m.group(3)}-{month_num}-{m.group(2).zfill(2)}"
    return None


def extract_amount(text: str) -> float | None:
    patterns = [
        r'(?:beløp|amount|monto|montant|Betrag|valor|sum|kr|NOK|EUR|USD)\s*:?\s*(\d[\d\s]*[.,]?\d*)',
        r'(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK|EUR|USD)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(' ', '').replace(',', '.')
            try:
                return float(val)
            except ValueError:
                pass
    return None


def extract_date_of_birth(text: str) -> str | None:
    patterns = [
        r'(?:fødselsdato|fødd|født|date of birth|fecha de nacimiento|date de naissance|Geburtsdatum|data de nascimento|birth\s*date)\s*:?\s*(\d{4}-\d{2}-\d{2})',
        r'(?:fødselsdato|fødd|født|date of birth|birth\s*date)\s*:?\s*(\d{1,2})[./](\d{1,2})[./](\d{4})',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            if len(m.groups()) == 3:
                return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
            return m.group(1)
    return None


def is_admin_request(text: str) -> bool:
    admin_terms = [
        "kontoadministrator", "administrator", "admin",
        "administrador", "administrateur",
        "admin role", "admin access", "admin-tilgang",
        "kontoadmin", "kontoadministratør",
    ]
    lower = text.lower()
    return any(t in lower for t in admin_terms)


def extract_address(text: str) -> dict | None:
    """Extract street address, zip, and city from prompt."""
    # Norwegian: "adresse Karl Johans gate 1, 0154 Oslo"
    # English: "address Karl Johans gate 1, 0154 Oslo"
    # Pattern: street + zip + city
    m = re.search(
        r'(?:adresse|address|dirección|adresse|Adresse|endereço|adresa)[:\s]+(.+?),?\s+(\d{4,5})\s+([A-ZÆØÅÄÖÜa-zæøåäöü]+(?:\s+[A-ZÆØÅÄÖÜa-zæøåäöü]+)*)',
        text, re.IGNORECASE
    )
    if m:
        return {
            "addressLine1": m.group(1).strip().rstrip(','),
            "postalCode": m.group(2),
            "city": m.group(3).strip(),
        }
    return None


def extract_payment_terms(text: str) -> int | None:
    """Extract payment terms in days."""
    m = re.search(r'(?:betalingsfrist|payment\s*terms?|plazo|délai|Zahlungsfrist|prazo)\s*:?\s*(\d+)\s*(?:dager|days|días|jours|Tage|dias)', text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_org_number(text: str) -> str | None:
    m = re.search(r'(?:org\.?\s*(?:nr|nummer|number)?\.?\s*:?\s*)(\d{9})', text, re.IGNORECASE)
    return m.group(1) if m else None


def extract_product_info(text: str) -> dict:
    info: dict[str, Any] = {}
    # Product name — try quoted first
    m = re.search(r'["\u201c\u201d«»]([^"\u201c\u201d«»]+)["\u201c\u201d«»]', text)
    if m:
        info["name"] = m.group(1).strip()
    else:
        name_pats = [
            # "produkt/product XYZ med/with ..." or "produkt XYZ, ..."
            r'(?:produkt|product|producto|produit|Produkt|produto)\s+(?:med\s+)?(?:navn\s+)?([A-ZÆØÅA-zæøå][\w\sæøåäöü&.\-]+?)(?=\s*(?:\.|,|med\b|with\b|nummer|number|pris|price|og\b|and\b|mit\b|preis|prix|precio|preço|$))',
            r'(?:navn|name|nombre|nom|Name|nome)\s+["\"]?([A-ZÆØÅA-zæøå][\w\sæøåäöü&.\-]+?)["\"]?\s*(?:og\b|and\b|,|\.|$)',
        ]
        for p in name_pats:
            m2 = re.search(p, text, re.IGNORECASE)
            if m2:
                info["name"] = m2.group(1).strip()
                break

    # Product number
    m = re.search(r'(?:produkt\s*nummer|product\s*number|Produktnummer|nr|number|número|numéro|numero)\s*:?\s*(\d+)', text, re.IGNORECASE)
    if m:
        info["number"] = m.group(1)

    # Price — multilingual with different formats
    price_pats = [
        r'(?:pris|price|precio|prix|Preis|preço)\s*(?:er|is|beträgt|est|es|é)?\s*:?\s*(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK|EUR|USD)?',
        r'(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK|EUR|USD)\s*(?:ekskl|excl|utan|ohne|sans|sin|sem|eks|ex|excluding)',
        r'(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK|EUR|USD)',
        r'(?:pris|price|precio|prix|Preis|preço)\s*:?\s*(\d[\d\s]*[.,]?\d*)',
    ]
    for p in price_pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(' ', '').replace(',', '.')
            try:
                info["price"] = float(val)
                break
            except ValueError:
                pass

    # VAT type extraction
    lower = text.lower()
    if any(w in lower for w in ['matvare', 'food', 'lebensmittel', 'alimentaire', 'aliment', '15 %', '15%']):
        info["vat_hint"] = "food"  # 15% rate
    elif any(w in lower for w in ['fritak', 'exempt', 'frei', 'exent', '0 %', '0%']):
        info["vat_hint"] = "exempt"

    return info


# ─── Task detection ─────────────────────────────────────────────────────────

def detect_task_type(prompt: str) -> str | None:
    lower = prompt.lower()

    # Detect COMPLEX multi-step tasks that should go straight to LLM
    complex_indicators = [
        # Timesheet + invoice (multi-step: needs both timesheet AND invoice)
        ("horas", "fatura"),  # Portuguese: hours + invoice
        ("timer", "faktura"),  # Norwegian: hours + invoice
        ("hours", "invoice"),  # English
        ("timesheet", "invoice"),
        ("stunden", "rechnung"),  # German
        ("heures", "facture"),  # French: hours + invoice
    ]
    for indicators in complex_indicators:
        if all(ind in lower for ind in indicators):
            return None  # Force LLM fallback

    # Special routing: order + invoice + payment combo → create_invoice (it handles payment inline)
    combo_invoice_keywords = [
        ("commande", "facture"),  # French: order → invoice
        ("bestilling", "faktura"),  # Norwegian
        ("order", "invoice"),  # English
        ("bestellung", "rechnung"),  # German
        ("pedido", "fatura"),  # Portuguese
        ("orden", "factura"),  # Spanish
    ]
    for combo_kw in combo_invoice_keywords:
        if all(k in lower for k in combo_kw):
            return "create_invoice"

    best_match = None
    best_score = 0
    for task_type, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lower:
                score = len(kw)
                if score > best_score:
                    best_score = score
                    best_match = task_type
    return best_match


# ─── Task executors ─────────────────────────────────────────────────────────

def exec_create_employee(prompt: str, client: TripletexClient) -> str:
    name = extract_name(prompt)
    email = extract_email(prompt)
    dob = extract_date_of_birth(prompt)
    is_admin = is_admin_request(prompt)
    phone = extract_phone(prompt)

    if not name:
        return "ERR: Could not extract employee name"

    # Need a department — get first available, create one if none exist
    dept = client.get("/department", params={"fields": "id", "count": "1"})
    dept_id = None
    if "values" in dept and dept["values"]:
        dept_id = dept["values"][0]["id"]
    else:
        # Create a default department
        dept_r = client.post("/department", json_body={"name": "Hovedavdeling", "departmentNumber": "1"})
        if "value" in dept_r:
            dept_id = dept_r["value"]["id"]

    # Build employee body
    body: dict[str, Any] = {
        "firstName": name[0],
        "lastName": name[1],
    }
    if dept_id:
        body["department"] = {"id": dept_id}

    # Determine userType based on email availability
    if email:
        body["email"] = email
        body["userType"] = "STANDARD"
    elif is_admin:
        # Admin requires login, so needs email — generate one
        body["email"] = f"{name[0].lower()}.{name[1].lower()}@company.no"
        body["userType"] = "STANDARD"
    else:
        # No email → NO_ACCESS (no login needed)
        body["userType"] = "NO_ACCESS"

    if dob:
        body["dateOfBirth"] = dob
    if phone:
        body["phoneNumberMobile"] = phone

    result = client.post("/employee", json_body=body)
    logger.info(f"Create employee: {result}")

    if "value" not in result:
        # Check if duplicate email error — try to find existing employee
        err_msg = str(result)
        if "allerede" in err_msg or "already" in err_msg or "duplicate" in err_msg.lower():
            # Search for existing employee
            search = {}
            if email:
                search["email"] = email
            elif name:
                search["firstName"] = name[0]
                search["lastName"] = name[1]
            search["fields"] = "id,firstName,lastName,email,version,allowInformationRegistration,department"
            search["count"] = "5"
            existing = client.get("/employee", params=search)
            if "values" in existing and existing["values"]:
                emp = existing["values"][0]
                emp_id = emp["id"]
                # Update admin if needed
                if is_admin:
                    emp["allowInformationRegistration"] = True
                    for key in list(emp.keys()):
                        if key in ("changes", "url", "displayName", "isContact", "isProxy"):
                            emp.pop(key, None)
                    client.put(f"/employee/{emp_id}", json_body=emp)
                return f"OK: Employee {name[0]} {name[1]} (ID: {emp_id})"
        # Try again with NO_ACCESS if STANDARD failed
        if body.get("userType") == "STANDARD":
            body["userType"] = "NO_ACCESS"
            body.pop("email", None)
            result = client.post("/employee", json_body=body)
            if "value" not in result:
                return f"ERR: {result}"

    if "value" not in result:
        return f"ERR: {result}"

    emp_id = result["value"]["id"]

    # Set admin flag if needed
    if is_admin:
        emp_data = result["value"].copy()
        emp_data["allowInformationRegistration"] = True
        for key in list(emp_data.keys()):
            if key in ("changes", "url", "displayName", "isContact", "isProxy"):
                del emp_data[key]
        try:
            client.put(f"/employee/{emp_id}", json_body=emp_data)
        except Exception as e:
            logger.warning(f"Admin setup failed: {e}")

    return f"OK: Employee {name[0]} {name[1]} (ID: {emp_id})"


def exec_create_customer(prompt: str, client: TripletexClient) -> str:
    company = extract_company_name(prompt)
    email = extract_email(prompt)
    phone = extract_phone(prompt)
    org_nr = extract_org_number(prompt)
    address = extract_address(prompt)
    payment_terms = extract_payment_terms(prompt)

    if not company:
        return "ERR: Could not extract customer name"

    body: dict[str, Any] = {"name": company, "isCustomer": True}
    if email:
        body["email"] = email
        body["invoiceEmail"] = email
    if phone:
        body["phoneNumber"] = phone
    if org_nr:
        body["organizationNumber"] = org_nr
    if payment_terms:
        body["invoicesDueIn"] = payment_terms
        body["invoicesDueInType"] = "DAYS"
    if address:
        body["physicalAddress"] = address

    result = client.post("/customer", json_body=body)
    logger.info(f"Create customer: {result}")
    if "value" not in result:
        return f"ERR: {result}"

    cust_id = result["value"]["id"]

    return f"OK: Customer {company} (ID: {cust_id})"


def exec_create_product(prompt: str, client: TripletexClient) -> str:
    info = extract_product_info(prompt)
    if not info.get("name"):
        m = re.search(r'["\u201c]([^"\u201d]+)["\u201d]', prompt)
        if m:
            info["name"] = m.group(1)
        else:
            return "ERR: Could not extract product name"

    body: dict[str, Any] = {"name": info["name"]}
    if "number" in info:
        body["number"] = info["number"]
    if "price" in info:
        body["priceExcludingVatCurrency"] = info["price"]
        body["priceIncludingVatCurrency"] = info["price"]

    # Set VAT type if hinted
    if info.get("vat_hint") == "food":
        # Look up food/low VAT rate
        vat_r = client.get("/ledger/vatType", params={"fields": "id,name,percentage", "count": "50"})
        if "values" in vat_r:
            for vt in vat_r["values"]:
                pct = vt.get("percentage", 0)
                if pct == 15 or (10 < pct < 16):
                    body["vatType"] = {"id": vt["id"]}
                    # Recalculate price including VAT
                    if "price" in info:
                        body["priceIncludingVatCurrency"] = round(info["price"] * (1 + pct / 100), 2)
                    break

    result = client.post("/product", json_body=body)
    logger.info(f"Create product: {result}")
    if "value" in result:
        return f"OK: Product {info['name']} (ID: {result['value']['id']})"
    return f"ERR: {result}"


def exec_create_department(prompt: str, client: TripletexClient) -> str:
    # Try quoted name first
    name = None
    m = re.search(r'["\u201c\u201d«»]([^"\u201c\u201d«»]+)["\u201c\u201d«»]', prompt)
    if m:
        name = m.group(1).strip()
    else:
        patterns = [
            r'(?:avdeling|department|abteilung|département|departamento)\s+(?:med\s+)?(?:navn\s+)?([A-ZÆØÅÄÖÜ][\w\sæøåäöü&.\-]+?)(?:\s*(?:\.|,|$|med\b|with\b|og\b|and\b|mit\b|nummer|number|nr))',
        ]
        for p in patterns:
            m2 = re.search(p, prompt, re.IGNORECASE)
            if m2:
                name = m2.group(1).strip()
                break
    if not name:
        return "ERR: Could not extract department name"

    m = re.search(r'(?:nummer|number|nr)\s*:?\s*(\d+)', prompt, re.IGNORECASE)
    body: dict[str, Any] = {"name": name}
    if m:
        body["departmentNumber"] = m.group(1)

    result = client.post("/department", json_body=body)
    logger.info(f"Create department: {result}")
    if "value" in result:
        return f"OK: Department {name} (ID: {result['value']['id']})"

    # If module error, enable department module and retry
    err_str = str(result)
    if "403" in err_str or "modul" in err_str.lower() or "module" in err_str.lower() or "tilgang" in err_str.lower():
        modules = client.get("/company/modules", params={"fields": "*"})
        if "value" in modules:
            mod_data = modules["value"]
            mod_data["moduleDepartment"] = True
            for key in list(mod_data.keys()):
                if key in ("changes", "url"):
                    del mod_data[key]
            client.put("/company/modules", json_body=mod_data)
            result = client.post("/department", json_body=body)
            if "value" in result:
                return f"OK: Department {name} (ID: {result['value']['id']})"

    return f"ERR: {result}"


def exec_create_project(prompt: str, client: TripletexClient) -> str:
    # Try quoted name first
    proj_name = None
    m = re.search(r'["\u201c\u201d«»]([^"\u201c\u201d«»]+)["\u201c\u201d«»]', prompt)
    if m:
        proj_name = m.group(1).strip()
    else:
        patterns = [
            r'(?:prosjekt|project|proyecto|projet|Projekt|projeto)\s+(?:med\s+)?(?:navn\s+)?([A-ZÆØÅÄÖÜ][\w\sæøåäöü&.\-]+?)(?:\s*(?:\.|,|$|for\b|knyttet|linked|tilknyttet|med\b|with\b|mit\b|og\b|and\b|und\b|nummer|number))',
        ]
        for p in patterns:
            m2 = re.search(p, prompt, re.IGNORECASE)
            if m2:
                proj_name = m2.group(1).strip()
                break
    if not proj_name:
        return "ERR: Could not extract project name"

    body: dict[str, Any] = {"name": proj_name}

    # Need a project manager
    emp = client.get("/employee", params={"fields": "id", "count": "1"})
    if "values" in emp and emp["values"]:
        body["projectManager"] = {"id": emp["values"][0]["id"]}

    # Link to customer if mentioned
    cust_name = extract_company_name(prompt)
    if cust_name:
        cr = client.get("/customer", params={"name": cust_name, "fields": "id", "count": "5"})
        if "values" in cr and cr["values"]:
            body["customer"] = {"id": cr["values"][0]["id"]}

    m = re.search(r'(?:prosjektnummer|project\s*number|nummer|number)\s*:?\s*(\d+)', prompt, re.IGNORECASE)
    if m:
        body["number"] = m.group(1)

    result = client.post("/project", json_body=body)
    logger.info(f"Create project: {result}")
    if "value" in result:
        return f"OK: Project {proj_name} (ID: {result['value']['id']})"

    # If 403 or module error, enable project module and retry
    err_str = str(result)
    if "403" in err_str or "modul" in err_str.lower() or "module" in err_str.lower() or "tilgang" in err_str.lower():
        modules = client.get("/company/modules", params={"fields": "*"})
        if "value" in modules:
            mod_data = modules["value"]
            mod_data["moduleProjectEconomy"] = True
            for key in list(mod_data.keys()):
                if key in ("changes", "url"):
                    del mod_data[key]
            client.put("/company/modules", json_body=mod_data)
            result = client.post("/project", json_body=body)
            if "value" in result:
                return f"OK: Project {proj_name} (ID: {result['value']['id']})"

    return f"ERR: {result}"


def _extract_product_lines(prompt: str) -> list[dict]:
    """Extract multiple product lines from prompt: 'Product Name (NUMBER) à PRICE NOK'"""
    lines: list[dict] = []
    # Pattern: "ProductName (PRODNUM) à/at/for/por PRICE NOK"
    product_line_pattern = re.findall(
        r'([A-ZÆØÅÄÖÜa-zæøåäöüéèêëàâáãçñíìîïóòôõúùûü][\w\sæøåäöüéèêëàâáãçñ\-]+?)\s*'
        r'\((\d+)\)\s*'
        r'(?:à|at|@|for|por|zu|a)\s*'
        r'(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK|EUR|USD)?',
        prompt, re.IGNORECASE
    )
    for name, number, price_str in product_line_pattern:
        try:
            price = float(price_str.replace(' ', '').replace(',', '.'))
            lines.append({"name": name.strip(), "number": number, "price": price})
        except ValueError:
            pass
    return lines


def _extract_org_number_any(text: str) -> str | None:
    """Extract organization number from various formats."""
    patterns = [
        r'(?:org\.?\s*(?:nr|nummer|number|n[°º]?)?\.?\s*:?\s*)(\d{9})',
        r'(?:n[°º]\s*org\.?\s*:?\s*)(\d{9})',  # French "n° org."
        r'(?:NIF|CIF|VAT|CNPJ|orgnr)\s*:?\s*(\d{9})',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def exec_create_invoice(prompt: str, client: TripletexClient) -> str:
    today = datetime.date.today().isoformat()
    cust_name = extract_company_name(prompt)
    amount = extract_amount(prompt)
    inv_date = extract_date(prompt) or today
    payment_terms = extract_payment_terms(prompt)
    org_nr = _extract_org_number_any(prompt) or extract_org_number(prompt)

    # Calculate due date
    if payment_terms:
        due_date = (datetime.date.fromisoformat(inv_date) + datetime.timedelta(days=payment_terms)).isoformat()
    else:
        due_date = (datetime.date.fromisoformat(inv_date) + datetime.timedelta(days=14)).isoformat()

    # Extract multi-product lines (e.g., "Licence logicielle (7127) à 6750 NOK")
    product_lines = _extract_product_lines(prompt)

    # Extract single product info
    product_m = re.search(r'(?:produkt|product|producto|produit|Produkt|produto)\s+["\u201c]?([^"\u201d,]+)["\u201d]?', prompt, re.IGNORECASE)

    # Extract quantity: "x10", "10 stk", "10 units", etc.
    qty_m = re.search(r'(?:x|×)\s*(\d+)|(\d+)\s*(?:stk|units?|pcs|enheter|unidades|unités|Stück|unidades)', prompt, re.IGNORECASE)
    qty = int(qty_m.group(1) or qty_m.group(2)) if qty_m else 1

    # Extract line description from prompt
    desc_m = re.search(r'(?:linje|line|línea|ligne|Zeile|linha|beskrivelse|description)[:\s]+(.+?)(?:,|\.|x\d|$)', prompt, re.IGNORECASE)
    description = desc_m.group(1).strip() if desc_m else "Fakturalinje"

    # Find or create customer
    cust_id = None
    if org_nr:
        cr = client.get("/customer", params={"organizationNumber": org_nr, "fields": "id,name", "count": "5"})
        if "values" in cr and cr["values"]:
            cust_id = cr["values"][0]["id"]
    if not cust_id and cust_name:
        cr = client.get("/customer", params={"name": cust_name, "fields": "id,name", "count": "5"})
        if "values" in cr and cr["values"]:
            cust_id = cr["values"][0]["id"]
    if not cust_id:
        cust_body: dict[str, Any] = {"name": cust_name or "Kunde", "isCustomer": True}
        if org_nr:
            cust_body["organizationNumber"] = org_nr
        cc = client.post("/customer", json_body=cust_body)
        if "value" in cc:
            cust_id = cc["value"]["id"]
    if not cust_id:
        return "ERR: Could not find/create customer"

    # Create order
    order = client.post("/order", json_body={
        "customer": {"id": cust_id},
        "orderDate": inv_date,
        "deliveryDate": inv_date,
    })
    if "value" not in order:
        return f"ERR: Order: {order}"
    order_id = order["value"]["id"]

    # Add order lines — multiple products if detected
    if product_lines:
        order_lines = []
        for pl in product_lines:
            ol: dict[str, Any] = {
                "order": {"id": order_id},
                "count": 1,
            }
            # Look up product by number
            pr = client.get("/product", params={"number": pl["number"], "fields": "id,name", "count": "1"})
            if "values" in pr and pr["values"]:
                ol["product"] = {"id": pr["values"][0]["id"]}
            else:
                # Create product
                pr_r = client.post("/product", json_body={
                    "name": pl["name"],
                    "number": pl["number"],
                    "priceExcludingVatCurrency": pl["price"],
                    "priceIncludingVatCurrency": pl["price"],
                })
                if "value" in pr_r:
                    ol["product"] = {"id": pr_r["value"]["id"]}
                else:
                    ol["description"] = pl["name"]
                    ol["unitPriceExcludingVatCurrency"] = pl["price"]
            order_lines.append(ol)

        if len(order_lines) == 1:
            client.post("/order/orderline", json_body=order_lines[0])
        elif order_lines:
            client.post("/order/orderline/list", json_body=order_lines)
    else:
        # Single product/line
        ol_body: dict[str, Any] = {
            "order": {"id": order_id},
            "count": qty,
        }
        # Find product by name
        product_id = None
        if product_m:
            pr = client.get("/product", params={"name": product_m.group(1).strip(), "fields": "id,name", "count": "5"})
            if "values" in pr and pr["values"]:
                product_id = pr["values"][0]["id"]
        if product_id:
            ol_body["product"] = {"id": product_id}
        else:
            ol_body["description"] = description
            ol_body["unitPriceExcludingVatCurrency"] = amount or 1000.0
        client.post("/order/orderline", json_body=ol_body)

    # Create invoice — use PUT /:invoice action endpoint with query params
    inv = client.put(f"/order/{order_id}/:invoice", params={
        "invoiceDate": inv_date,
        "sendToCustomer": "false",
    })
    if "value" in inv:
        inv_id = inv["value"]["id"]

        # Register payment if requested in the same prompt
        lower = prompt.lower()
        payment_keywords = [
            "regist", "payment", "betaling", "innbetaling", "betal",
            "pagamento", "pago", "paiement", "zahlung", "bezahl",
        ]
        if any(kw in lower for kw in payment_keywords):
            # Get payment type
            pt = client.get("/invoice/paymentType", params={"fields": "id,description", "count": "10"})
            pt_id = 0
            if "values" in pt and pt["values"]:
                pt_id = pt["values"][0]["id"]
            # Get invoice amount for full payment
            inv_detail = client.get(f"/invoice/{inv_id}", params={"fields": "id,amount,amountCurrency"})
            pay_amount = amount or 0
            if "value" in inv_detail:
                pay_amount = inv_detail["value"].get("amount", inv_detail["value"].get("amountCurrency", pay_amount))
            if pay_amount:
                client.put(f"/invoice/{inv_id}/:payment", params={
                    "paymentDate": inv_date,
                    "paymentTypeId": str(pt_id),
                    "paidAmount": str(pay_amount),
                    "paidAmountCurrency": str(pay_amount),
                })

        return f"OK: Invoice (ID: {inv_id})"
    return f"ERR: Invoice: {inv}"


def exec_create_supplier(prompt: str, client: TripletexClient) -> str:
    company = extract_company_name(prompt)
    email = extract_email(prompt)
    org_nr = extract_org_number(prompt)
    phone = extract_phone(prompt)
    address = extract_address(prompt)
    if not company:
        return "ERR: Could not extract supplier name"
    body: dict[str, Any] = {"name": company, "isSupplier": True}
    if email:
        body["email"] = email
    if org_nr:
        body["organizationNumber"] = org_nr
    if phone:
        body["phoneNumber"] = phone
    result = client.post("/supplier", json_body=body)
    if "value" not in result:
        return f"ERR: {result}"
    sup_id = result["value"]["id"]
    if address:
        postal_addr = result["value"].get("postalAddress")
        if postal_addr and postal_addr.get("id"):
            client.put(f"/address/{postal_addr['id']}", json_body=address)
    return f"OK: Supplier {company} (ID: {sup_id})"


def exec_delete_travel_expense(prompt: str, client: TripletexClient) -> str:
    r = client.get("/travelExpense", params={"fields": "id,title", "count": "100"})
    if "values" not in r or not r["values"]:
        return "OK: No travel expenses to delete"
    for te in r["values"]:
        client.delete(f"/travelExpense/{te['id']}")
    return f"OK: Deleted {len(r['values'])} travel expense(s)"


def exec_register_payment(prompt: str, client: TripletexClient) -> str:
    """Register a payment on an invoice."""
    today = datetime.date.today().isoformat()
    amount = extract_amount(prompt)
    pay_date = extract_date(prompt) or today

    # Find invoice — search by number if mentioned
    inv_num = re.search(r'(?:faktura|invoice)\s*(?:nummer|number|nr|#)?\s*:?\s*(\d+)', prompt, re.IGNORECASE)

    invoices = None
    if inv_num:
        invoices = client.get("/invoice", params={
            "invoiceNumber": inv_num.group(1),
            "fields": "id,invoiceNumber,amount,amountCurrency,amountOutstanding",
            "count": "5"
        })
    if not invoices or "values" not in invoices or not invoices["values"]:
        # Get all invoices
        invoices = client.get("/invoice", params={
            "fields": "id,invoiceNumber,amount,amountCurrency,amountOutstanding",
            "count": "10"
        })

    if "values" not in invoices or not invoices["values"]:
        return "ERR: No invoices found"

    inv = invoices["values"][0]
    inv_id = inv["id"]
    pay_amount = amount or inv.get("amountOutstanding", inv.get("amount", 0))

    # Get payment type
    pt = client.get("/invoice/paymentType", params={"fields": "id,description", "count": "10"})
    pt_id = 0
    if "values" in pt and pt["values"]:
        pt_id = pt["values"][0]["id"]

    # Register payment — action endpoints use PUT with QUERY PARAMS, not body
    result = client.put(f"/invoice/{inv_id}/:payment", params={
        "paymentDate": pay_date,
        "paymentTypeId": str(pt_id),
        "paidAmount": str(pay_amount),
        "paidAmountCurrency": str(pay_amount),
    })
    logger.info(f"Register payment: {result}")

    if "value" in result or result.get("status") == 204:
        return f"OK: Payment registered on invoice {inv_id}"

    return f"ERR: {result}"


def exec_update_employee(prompt: str, client: TripletexClient) -> str:
    """Update an existing employee."""
    name = extract_name(prompt)
    email = extract_email(prompt)
    phone = extract_phone(prompt)

    # Find the employee
    search_params: dict[str, str] = {"fields": "id,firstName,lastName,email,phoneNumberMobile,userType,allowInformationRegistration,department,version", "count": "10"}
    if name:
        search_params["firstName"] = name[0]
        search_params["lastName"] = name[1]

    emp_result = client.get("/employee", params=search_params)
    if "values" not in emp_result or not emp_result["values"]:
        return "ERR: Employee not found"

    emp = emp_result["values"][0]
    emp_id = emp["id"]

    # Apply updates
    if email:
        emp["email"] = email
    if phone:
        emp["phoneNumberMobile"] = phone
    if is_admin_request(prompt):
        emp["allowInformationRegistration"] = True

    # Clean read-only fields
    for key in list(emp.keys()):
        if key in ("changes", "url", "displayName", "isContact", "isProxy"):
            del emp[key]

    result = client.put(f"/employee/{emp_id}", json_body=emp)
    if "value" in result:
        return f"OK: Employee updated (ID: {emp_id})"
    return f"ERR: {result}"


def exec_update_customer(prompt: str, client: TripletexClient) -> str:
    """Update an existing customer."""
    company = extract_company_name(prompt)
    email = extract_email(prompt)
    phone = extract_phone(prompt)

    search_params: dict[str, str] = {"fields": "id,name,email,phoneNumber,organizationNumber,invoicesDueIn,postalAddress,version", "count": "10"}
    if company:
        search_params["name"] = company

    cust_result = client.get("/customer", params=search_params)
    if "values" not in cust_result or not cust_result["values"]:
        return "ERR: Customer not found"

    cust = cust_result["values"][0]
    cust_id = cust["id"]

    if email:
        cust["email"] = email
    if phone:
        cust["phoneNumber"] = phone

    # Clean read-only fields
    for key in list(cust.keys()):
        if key in ("changes", "url"):
            del cust[key]

    result = client.put(f"/customer/{cust_id}", json_body=cust)
    if "value" in result:
        return f"OK: Customer updated (ID: {cust_id})"
    return f"ERR: {result}"


def exec_create_contact(prompt: str, client: TripletexClient) -> str:
    """Create a contact person for a customer."""
    name = extract_name(prompt)
    email = extract_email(prompt)
    phone = extract_phone(prompt)
    company = extract_company_name(prompt)

    if not name:
        return "ERR: Could not extract contact name"

    # Find customer
    cust_id = None
    if company:
        cr = client.get("/customer", params={"name": company, "fields": "id", "count": "5"})
        if "values" in cr and cr["values"]:
            cust_id = cr["values"][0]["id"]

    body: dict[str, Any] = {
        "firstName": name[0],
        "lastName": name[1],
    }
    if email:
        body["email"] = email
    if phone:
        body["phoneNumber"] = phone
    if cust_id:
        body["customer"] = {"id": cust_id}

    result = client.post("/contact", json_body=body)
    if "value" in result:
        return f"OK: Contact {name[0]} {name[1]} (ID: {result['value']['id']})"
    return f"ERR: {result}"


def exec_create_travel_expense(prompt: str, client: TripletexClient) -> str:
    today = datetime.date.today().isoformat()

    # Find employee — by name if specified, otherwise first available
    name = extract_name(prompt)
    email = extract_email(prompt)
    emp_params: dict[str, str] = {"fields": "id,firstName,lastName", "count": "10"}
    if email:
        emp_params["email"] = email
    elif name:
        emp_params["firstName"] = name[0]
        emp_params["lastName"] = name[1]
    emp = client.get("/employee", params=emp_params)
    if "values" not in emp or not emp["values"]:
        # Try without name filter
        emp = client.get("/employee", params={"fields": "id", "count": "1"})
    if "values" not in emp or not emp["values"]:
        # Create the employee if name is known
        if name:
            dept = client.get("/department", params={"fields": "id", "count": "1"})
            dept_id = None
            if "values" in dept and dept["values"]:
                dept_id = dept["values"][0]["id"]
            else:
                dept_r = client.post("/department", json_body={"name": "Hovedavdeling", "departmentNumber": "1"})
                if "value" in dept_r:
                    dept_id = dept_r["value"]["id"]
            emp_body: dict[str, Any] = {"firstName": name[0], "lastName": name[1], "userType": "NO_ACCESS"}
            if email:
                emp_body["email"] = email
                emp_body["userType"] = "STANDARD"
            if dept_id:
                emp_body["department"] = {"id": dept_id}
            emp_r = client.post("/employee", json_body=emp_body)
            if "value" in emp_r:
                emp = {"values": [emp_r["value"]]}
            else:
                return f"ERR: Could not create employee: {emp_r}"
        else:
            return "ERR: No employee found"
    emp_id = emp["values"][0]["id"]

    # Extract title — try quoted first, then keyword pattern
    title = "Reiseregning"
    title_m = re.search(r'["\u201c\u201d«»]([^"\u201c\u201d«»]+)["\u201c\u201d«»]', prompt)
    if title_m:
        title = title_m.group(1).strip()
    else:
        title_m2 = re.search(r'(?:tittel|title|título|titre|Titel)\s*:?\s*(.+?)(?:\.|,|$)', prompt, re.IGNORECASE)
        if title_m2:
            title = title_m2.group(1).strip()

    # Extract date
    travel_date = extract_date(prompt) or today

    result = client.post("/travelExpense", json_body={
        "employee": {"id": emp_id},
        "title": title,
    })
    if "value" not in result:
        return f"ERR: {result}"

    te_id = result["value"]["id"]

    # Lookup cost categories, vatType, currency, and paymentType once
    cats = client.get("/travelExpense/costCategory", params={"fields": "id,name", "count": "50"})
    cat_map: dict[str, int] = {}
    default_cat_id = None
    if "values" in cats and cats["values"]:
        for c in cats["values"]:
            cname = c.get("name", "").lower()
            cat_map[cname] = c["id"]
            if default_cat_id is None:
                default_cat_id = c["id"]

    # Get available travel payment types
    pay_types = client.get("/travelExpense/paymentType", params={"fields": "id,description", "count": "10"})
    pay_type_id = None
    if "values" in pay_types and pay_types["values"]:
        pay_type_id = pay_types["values"][0]["id"]

    # VAT type — usually 0% for expenses
    vat_types = client.get("/ledger/vatType", params={"fields": "id,name,percentage", "count": "50"})
    vat_0_id = None
    vat_25_id = None
    if "values" in vat_types and vat_types["values"]:
        for vt in vat_types["values"]:
            pct = vt.get("percentage", -1)
            if pct == 0 and vat_0_id is None:
                vat_0_id = vt["id"]
            if pct == 25 and vat_25_id is None:
                vat_25_id = vt["id"]

    # Currency
    curr = client.get("/currency", params={"code": "NOK", "fields": "id", "count": "1"})
    curr_id = 1  # default NOK
    if "values" in curr and curr["values"]:
        curr_id = curr["values"][0]["id"]

    # Parse all expenses from the prompt
    lower = prompt.lower()
    cost_lines: list[dict[str, Any]] = []

    # Pattern 1: Per diem / diett — "N dager med diett (dagsats X kr)"
    diett_m = re.search(r'(\d+)\s*(?:dager|days|dias|jours|Tage)\s+(?:med\s+)?(?:diett|per\s*diem|dieta|indemnité)', prompt, re.IGNORECASE)
    diett_rate_m = re.search(r'(?:dagsats|daily\s*rate|tarifa\s*diaria|taux\s*journalier|Tagessatz)\s*:?\s*(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK)?', prompt, re.IGNORECASE)
    if diett_m:
        days = int(diett_m.group(1))
        rate = 800.0  # default
        if diett_rate_m:
            rate = float(diett_rate_m.group(1).replace(' ', '').replace(',', '.'))
        # Find per diem category
        cat_id = default_cat_id
        for cname, cid in cat_map.items():
            if any(w in cname for w in ['diett', 'kost', 'diet', 'per diem', 'diem', 'dag']):
                cat_id = cid
                break
        cost_lines.append({"desc": "Diett", "count": days, "rate": rate, "amount": days * rate, "cat_id": cat_id})

    # Pattern 2: Individual expense items — "utlegg: X kr og Y kr" or "flybillett 7150 kr og taxi 450 kr"
    expense_patterns = [
        # Norwegian
        (r'(?:fly(?:billett)?|flight|flug|vol|vuelo|voo)\s*:?\s*(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK)', ['fly', 'reise']),
        (r'(?:taxi|drosje|cab|Taxi)\s*:?\s*(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK)', ['taxi', 'transport', 'reise']),
        (r'(?:hotell|hotel|Hotel|hôtel)\s*:?\s*(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK)', ['hotell', 'overnatting', 'hotel']),
        (r'(?:tog|train|Zug|tren|trem)\s*:?\s*(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK)', ['tog', 'transport', 'reise']),
        (r'(?:buss|bus|Bus|autobus)\s*:?\s*(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK)', ['buss', 'transport', 'reise']),
        (r'(?:parkering|parking|Parkplatz|stationnement)\s*:?\s*(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK)', ['parkering', 'transport']),
        (r'(?:mat|food|Essen|repas|comida|refeição)\s*:?\s*(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK)', ['mat', 'kost', 'representasjon']),
        (r'(?:km|kilometer|mileage)\s*:?\s*(\d[\d\s]*[.,]?\d*)', ['km', 'kjøregodtgjørelse', 'reise']),
    ]

    for pattern, cat_keywords in expense_patterns:
        m = re.search(pattern, prompt, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(' ', '').replace(',', '.'))
            cat_id = default_cat_id
            for kw in cat_keywords:
                for cname, cid in cat_map.items():
                    if kw in cname:
                        cat_id = cid
                        break
                if cat_id != default_cat_id:
                    break
            desc = pattern.split('|')[0].split('(?:')[1].split(')')[0] if '(?:' in pattern else "Utlegg"
            cost_lines.append({"desc": desc, "count": 1, "rate": val, "amount": val, "cat_id": cat_id})

    # Pattern 3: Generic amount if no specific expenses found
    if not cost_lines:
        amount = extract_amount(prompt)
        if amount:
            cost_lines.append({"desc": "Utlegg", "count": 1, "rate": amount, "amount": amount, "cat_id": default_cat_id})

    # Create cost lines
    vat_id = vat_0_id or (vat_25_id or 0)
    for cl in cost_lines:
        cost_body: dict[str, Any] = {
            "travelExpense": {"id": te_id},
            "currency": {"id": curr_id},
            "date": travel_date,
            "count": cl["count"],
            "rate": cl["rate"],
            "amount": cl["amount"],
            "paymentType": "EMPLOYEE",
        }
        if cl.get("cat_id"):
            cost_body["costCategory"] = {"id": cl["cat_id"]}
        if vat_id:
            cost_body["vatType"] = {"id": vat_id}
        cr = client.post("/travelExpense/cost", json_body=cost_body)
        logger.info(f"Travel cost line: {cr}")

    return f"OK: Travel expense (ID: {te_id}) with {len(cost_lines)} cost line(s)"


def exec_create_credit_note(prompt: str, client: TripletexClient) -> str:
    """Create a credit note for an invoice. Uses PUT /:createCreditNote action endpoint."""
    inv_num = re.search(r'(?:faktura|invoice|rechnung|facture|factura|fatura)\s*(?:nummer|number|nr|#|no\.?)?\s*:?\s*(\d+)', prompt, re.IGNORECASE)

    invoices = None
    if inv_num:
        invoices = client.get("/invoice", params={
            "invoiceNumber": inv_num.group(1),
            "fields": "id,invoiceNumber",
            "count": "5"
        })
    if not invoices or "values" not in invoices or not invoices["values"]:
        # Get all invoices
        invoices = client.get("/invoice", params={
            "fields": "id,invoiceNumber",
            "count": "10"
        })
    if "values" not in invoices or not invoices["values"]:
        return "ERR: No invoices found for credit note"

    inv_id = invoices["values"][0]["id"]

    # Action endpoint uses PUT, not POST
    result = client.put(f"/invoice/{inv_id}/:createCreditNote", json_body={})
    logger.info(f"Create credit note: {result}")

    if "value" in result:
        return f"OK: Credit note created for invoice {inv_id}"
    return f"ERR: {result}"


def exec_delete_voucher(prompt: str, client: TripletexClient) -> str:
    """Delete vouchers from the ledger."""
    vouchers = client.get("/ledger/voucher", params={"fields": "id", "count": "100"})
    if "values" not in vouchers or not vouchers["values"]:
        return "OK: No vouchers to delete"
    deleted = 0
    for v in vouchers["values"]:
        client.delete(f"/ledger/voucher/{v['id']}")
        deleted += 1
    return f"OK: Deleted {deleted} voucher(s)"


def exec_send_invoice(prompt: str, client: TripletexClient) -> str:
    """Send an invoice (mark as sent/delivered)."""
    inv_num = re.search(r'(?:faktura|invoice|rechnung|facture|factura|fatura)\s*(?:nummer|number|nr|#|no\.?)?\s*:?\s*(\d+)', prompt, re.IGNORECASE)

    invoices = None
    if inv_num:
        invoices = client.get("/invoice", params={
            "invoiceNumber": inv_num.group(1),
            "fields": "id,invoiceNumber",
            "count": "5"
        })
    if not invoices or "values" not in invoices or not invoices["values"]:
        invoices = client.get("/invoice", params={"fields": "id,invoiceNumber", "count": "10"})

    if "values" not in invoices or not invoices["values"]:
        return "ERR: No invoices found to send"

    inv_id = invoices["values"][0]["id"]

    # Action endpoint: PUT /:send
    result = client.put(f"/invoice/{inv_id}/:send", json_body={
        "sendType": "EMAIL",
    })
    logger.info(f"Send invoice: {result}")

    if "value" in result or result.get("status") == 204:
        return f"OK: Invoice {inv_id} sent"

    return f"ERR: {result}"


def exec_approve_travel_expense(prompt: str, client: TripletexClient) -> str:
    """Approve travel expense(s)."""
    te_list = client.get("/travelExpense", params={"fields": "id,title,status", "count": "100"})
    if "values" not in te_list or not te_list["values"]:
        return "ERR: No travel expenses found"

    approved = 0
    for te in te_list["values"]:
        te_id = te["id"]
        result = client.put(f"/travelExpense/{te_id}/:approve", json_body={})
        logger.info(f"Approve travel expense {te_id}: {result}")
        if "value" in result or result.get("status") == 204 or not result.get("error"):
            approved += 1

    if approved > 0:
        return f"OK: Approved {approved} travel expense(s)"
    return "ERR: Could not approve travel expenses"


def exec_deliver_travel_expense(prompt: str, client: TripletexClient) -> str:
    """Deliver/submit travel expense(s)."""
    te_list = client.get("/travelExpense", params={"fields": "id,title,status", "count": "100"})
    if "values" not in te_list or not te_list["values"]:
        return "ERR: No travel expenses found"

    delivered = 0
    for te in te_list["values"]:
        te_id = te["id"]
        result = client.put(f"/travelExpense/{te_id}/:deliver", json_body={})
        logger.info(f"Deliver travel expense {te_id}: {result}")
        if "value" in result or result.get("status") == 204 or not result.get("error"):
            delivered += 1

    if delivered > 0:
        return f"OK: Delivered {delivered} travel expense(s)"
    return "ERR: Could not deliver travel expenses"


def exec_create_timesheet(prompt: str, client: TripletexClient) -> str:
    """Register timesheet / hours for an employee on a project."""
    name = extract_name(prompt)
    email = extract_email(prompt)
    date = extract_date(prompt) or datetime.date.today().isoformat()

    # Extract hours
    hours_m = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:timer|hours|horas|heures|Stunden|stunder|timar)', prompt, re.IGNORECASE)
    hours = float(hours_m.group(1).replace(',', '.')) if hours_m else None
    if not hours:
        hours_m2 = re.search(r'(?:timer|hours|horas|heures|Stunden)\s*:?\s*(\d+(?:[.,]\d+)?)', prompt, re.IGNORECASE)
        hours = float(hours_m2.group(1).replace(',', '.')) if hours_m2 else None
    if not hours:
        return "ERR: Could not extract hours"

    # Find employee
    emp_params: dict[str, str] = {"fields": "id,firstName,lastName", "count": "10"}
    if email:
        emp_params["email"] = email
    elif name:
        emp_params["firstName"] = name[0]
        emp_params["lastName"] = name[1]
    emp = client.get("/employee", params=emp_params)
    if "values" not in emp or not emp["values"]:
        emp = client.get("/employee", params={"fields": "id", "count": "1"})
    if "values" not in emp or not emp["values"]:
        return "ERR: No employee found"
    emp_id = emp["values"][0]["id"]

    # Find or create project
    proj_name = None
    proj_m = re.search(r'(?:prosjekt|project|proyecto|projet|Projekt|projeto)\s+["\u201c]?([^"\u201d,]+?)["\u201d]?(?:\s*(?:\.|,|$|for\b|med\b|with\b))', prompt, re.IGNORECASE)
    if not proj_m:
        proj_m = re.search(r'["\u201c\u201d«»]([^"\u201c\u201d«»]+)["\u201c\u201d«»]', prompt)
    if proj_m:
        proj_name = proj_m.group(1).strip()

    proj_id = None
    if proj_name:
        pr = client.get("/project", params={"name": proj_name, "fields": "id", "count": "5"})
        if "values" in pr and pr["values"]:
            proj_id = pr["values"][0]["id"]
    if not proj_id:
        # Get first project or create one
        pr = client.get("/project", params={"fields": "id", "count": "1"})
        if "values" in pr and pr["values"]:
            proj_id = pr["values"][0]["id"]
        else:
            proj_body: dict[str, Any] = {"name": proj_name or "Standard", "projectManager": {"id": emp_id}}
            pr_r = client.post("/project", json_body=proj_body)
            if "value" in pr_r:
                proj_id = pr_r["value"]["id"]

    if not proj_id:
        return "ERR: Could not find/create project"

    # Find or create activity
    act_name = None
    act_m = re.search(r'(?:aktivitet|activity|actividad|activité|Aktivität|atividade)\s+["\u201c]?([^"\u201d,]+)', prompt, re.IGNORECASE)
    if act_m:
        act_name = act_m.group(1).strip()

    act_id = None
    acts = client.get("/activity", params={"fields": "id,name", "count": "10"})
    if "values" in acts and acts["values"]:
        if act_name:
            for a in acts["values"]:
                if act_name.lower() in a.get("name", "").lower():
                    act_id = a["id"]
                    break
        if not act_id:
            act_id = acts["values"][0]["id"]
    if not act_id:
        act_r = client.post("/activity", json_body={"name": act_name or "Arbeid"})
        if "value" in act_r:
            act_id = act_r["value"]["id"]

    if not act_id:
        return "ERR: Could not find/create activity"

    # Extract comment
    comment_m = re.search(r'(?:kommentar|comment|comentario|commentaire|Kommentar)\s*:?\s*["\"]?(.+?)["\"]?(?:\.|,|$)', prompt, re.IGNORECASE)
    comment = comment_m.group(1).strip() if comment_m else None

    # Create timesheet entry
    body: dict[str, Any] = {
        "employee": {"id": emp_id},
        "project": {"id": proj_id},
        "activity": {"id": act_id},
        "date": date,
        "hours": hours,
    }
    if comment:
        body["comment"] = comment

    result = client.post("/timesheet/entry", json_body=body)
    logger.info(f"Create timesheet: {result}")
    if "value" in result:
        return f"OK: Timesheet entry ({hours}h) created (ID: {result['value']['id']})"
    return f"ERR: {result}"


def exec_create_voucher(prompt: str, client: TripletexClient) -> str:
    """Create a journal entry / voucher."""
    today = datetime.date.today().isoformat()
    date = extract_date(prompt) or today

    # Extract description
    desc_m = re.search(r'(?:beskrivelse|description|descripción|Beschreibung|descrição)\s*:?\s*["\"]?(.+?)["\"]?(?:\.|,|$)', prompt, re.IGNORECASE)
    description = desc_m.group(1).strip() if desc_m else "Bilag"

    # Extract account numbers and amounts
    # Pattern: "debet konto 1920 1000 kr, kredit konto 3000 1000 kr"
    postings: list[dict[str, Any]] = []

    # Look for debit/credit pairs
    debit_m = re.search(r'(?:debet|debit|débito|débit|Soll|debe)\s*(?:konto|account|cuenta|compte|Konto|conta)?\s*:?\s*(\d{4})\s*(?:med\s+|for\s+)?(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK)?', prompt, re.IGNORECASE)
    credit_m = re.search(r'(?:kredit|credit|crédito|crédit|Haben|crédito)\s*(?:konto|account|cuenta|compte|Konto|conta)?\s*:?\s*(\d{4})\s*(?:med\s+|for\s+)?(\d[\d\s]*[.,]?\d*)\s*(?:kr|NOK)?', prompt, re.IGNORECASE)

    if debit_m and credit_m:
        debit_acct = debit_m.group(1)
        debit_amt = float(debit_m.group(2).replace(' ', '').replace(',', '.'))
        credit_acct = credit_m.group(1)
        credit_amt = float(credit_m.group(2).replace(' ', '').replace(',', '.'))

        # Look up account IDs
        da = client.get("/ledger/account", params={"number": debit_acct, "fields": "id,number", "count": "1"})
        ca = client.get("/ledger/account", params={"number": credit_acct, "fields": "id,number", "count": "1"})

        if "values" in da and da["values"] and "values" in ca and ca["values"]:
            postings.append({"date": date, "account": {"id": da["values"][0]["id"]}, "amount": debit_amt})
            postings.append({"date": date, "account": {"id": ca["values"][0]["id"]}, "amount": -credit_amt})

    # Alternative: look for account number patterns like "konto 1920" and amount
    if not postings:
        acct_matches = re.findall(r'(?:konto|account|cuenta|compte|Konto|conta)\s*:?\s*(\d{4})', prompt, re.IGNORECASE)
        amount = extract_amount(prompt)

        if len(acct_matches) >= 2 and amount:
            a1 = client.get("/ledger/account", params={"number": acct_matches[0], "fields": "id", "count": "1"})
            a2 = client.get("/ledger/account", params={"number": acct_matches[1], "fields": "id", "count": "1"})
            if "values" in a1 and a1["values"] and "values" in a2 and a2["values"]:
                postings.append({"date": date, "account": {"id": a1["values"][0]["id"]}, "amount": amount})
                postings.append({"date": date, "account": {"id": a2["values"][0]["id"]}, "amount": -amount})

    if not postings:
        return "ERR: Could not extract account numbers and amounts for voucher"

    # Look up voucher type for manual journal entries
    voucher_body: dict[str, Any] = {
        "date": date,
        "description": description,
        "postings": postings,
    }
    vt_r = client.get("/ledger/voucherType", params={"fields": "id,name", "count": "20"})
    if "values" in vt_r and vt_r["values"]:
        for vt in vt_r["values"]:
            name_l = vt.get("name", "").lower()
            if any(w in name_l for w in ["manuelt", "manual", "diverse", "journal"]):
                voucher_body["type"] = {"id": vt["id"]}
                break
        if "type" not in voucher_body:
            voucher_body["type"] = {"id": vt_r["values"][0]["id"]}

    result = client.post("/ledger/voucher", json_body=voucher_body)
    logger.info(f"Create voucher: {result}")
    if "value" in result:
        return f"OK: Voucher created (ID: {result['value']['id']})"

    # Retry without type if it failed
    if "type" in voucher_body:
        del voucher_body["type"]
        result2 = client.post("/ledger/voucher", json_body=voucher_body)
        if "value" in result2:
            return f"OK: Voucher created (ID: {result2['value']['id']})"

    return f"ERR: {result}"


def exec_enable_module(prompt: str, client: TripletexClient) -> str:
    """Enable a company module."""
    lower = prompt.lower()

    # Determine which module to enable
    module_field = None
    if any(w in lower for w in ['avdeling', 'department', 'abteilung', 'département', 'departamento']):
        module_field = "moduleDepartment"
    elif any(w in lower for w in ['prosjekt', 'project', 'proyecto', 'projet', 'projekt', 'projeto']):
        module_field = "moduleProjectEconomy"
    elif any(w in lower for w in ['dimensjon', 'dimension', 'dimensión', 'dimensão']):
        module_field = "moduleCustomDimension"
    elif any(w in lower for w in ['ansatt', 'employee', 'mitarbeiter', 'employé', 'empleado', 'funcionário']):
        module_field = "moduleEmployee"
    elif any(w in lower for w in ['produkt', 'product', 'producto', 'produit', 'produto']):
        module_field = "moduleProduct"
    elif any(w in lower for w in ['faktura', 'invoice', 'rechnung', 'facture', 'factura', 'fatura']):
        module_field = "moduleInvoice"
    elif any(w in lower for w in ['kunde', 'customer', 'client', 'Kunde', 'cliente']):
        module_field = "moduleCustomer"

    if not module_field:
        return "ERR: Could not determine which module to enable"

    # GET current modules, modify, PUT back
    modules = client.get("/company/modules", params={"fields": "*"})
    if "value" not in modules:
        return f"ERR: Could not get modules: {modules}"

    mod_data = modules["value"]
    mod_data[module_field] = True

    # Clean read-only fields
    for key in list(mod_data.keys()):
        if key in ("changes", "url"):
            del mod_data[key]

    result = client.put("/company/modules", json_body=mod_data)
    logger.info(f"Enable module {module_field}: {result}")
    if "value" in result:
        return f"OK: Module {module_field} enabled"
    return f"ERR: {result}"


def exec_create_supplier_invoice(prompt: str, client: TripletexClient) -> str:
    """Register a supplier/purchase invoice: create supplier → voucher with deductible VAT."""
    today = datetime.date.today().isoformat()
    date = extract_date(prompt) or today

    # Extract details
    company = extract_company_name(prompt)
    org_nr = _extract_org_number_any(prompt) or extract_org_number(prompt)
    amount = extract_amount(prompt)

    # Extract account number for expense account
    acct_m = re.search(r'(?:konto|account|compte|cuenta|conta|Konto)\s*:?\s*(\d{4})', prompt, re.IGNORECASE)
    expense_acct_num = acct_m.group(1) if acct_m else None

    # Extract invoice reference (INV-2026-XXXX etc.)
    inv_ref_m = re.search(r'(?:faktura|invoice|fatura|facture|Rechnung)\s*(?:nr\.?|nummer|number|n[°º]?)\s*:?\s*([A-Za-z0-9\-]+)', prompt, re.IGNORECASE)
    if not inv_ref_m:
        inv_ref_m = re.search(r'(INV-\d{4}-\d+)', prompt)
    inv_ref = inv_ref_m.group(1) if inv_ref_m else None

    # Extract VAT percentage
    vat_pct_m = re.search(r'(?:mva|iva|vat|tva|mwst|ust)\s*(?:d[eé]ductible|inkludert|inclu[sí]d[oa]?)?\s*(?:correct[oa]?|korrekt)?\s*\(?\s*(\d{1,2})\s*%', prompt, re.IGNORECASE)
    if not vat_pct_m:
        vat_pct_m = re.search(r'(\d{1,2})\s*%\s*(?:mva|iva|vat|tva|mwst)', prompt, re.IGNORECASE)
    vat_pct = int(vat_pct_m.group(1)) if vat_pct_m else 25

    if not company:
        return "ERR: Could not extract supplier name"
    if not amount:
        return "ERR: Could not extract amount"

    # Step 1: Find or create supplier
    supplier_id = None
    if org_nr:
        sr = client.get("/supplier", params={"organizationNumber": org_nr, "fields": "id,name", "count": "5"})
        if "values" in sr and sr["values"]:
            supplier_id = sr["values"][0]["id"]
    if not supplier_id:
        sr = client.get("/supplier", params={"name": company, "fields": "id,name", "count": "5"})
        if "values" in sr and sr["values"]:
            supplier_id = sr["values"][0]["id"]
    if not supplier_id:
        sup_body: dict[str, Any] = {"name": company, "isSupplier": True}
        if org_nr:
            sup_body["organizationNumber"] = org_nr
        sr = client.post("/supplier", json_body=sup_body)
        if "value" in sr:
            supplier_id = sr["value"]["id"]
    if not supplier_id:
        return f"ERR: Could not create supplier {company}"

    # Step 2: Look up accounts
    expense_acct_id = None
    if expense_acct_num:
        ar = client.get("/ledger/account", params={"number": expense_acct_num, "fields": "id", "count": "1"})
        if "values" in ar and ar["values"]:
            expense_acct_id = ar["values"][0]["id"]

    # Supplier ledger account (leverandørgjeld) — usually 2400
    supplier_acct_id = None
    for acct_num in ["2400", "2401", "2000"]:
        ar = client.get("/ledger/account", params={"number": acct_num, "fields": "id", "count": "1"})
        if "values" in ar and ar["values"]:
            supplier_acct_id = ar["values"][0]["id"]
            break

    # Input VAT account — usually 2710 (25%), 2711 (15%), 2714 (8%)
    vat_acct_id = None
    vat_acct_candidates = {"25": ["2710", "2711"], "15": ["2711", "2714"], "12": ["2714"], "8": ["2714"]}
    for va in vat_acct_candidates.get(str(vat_pct), ["2710"]):
        ar = client.get("/ledger/account", params={"number": va, "fields": "id", "count": "1"})
        if "values" in ar and ar["values"]:
            vat_acct_id = ar["values"][0]["id"]
            break

    if not expense_acct_id or not supplier_acct_id:
        return "ERR: Could not find required accounts"

    # Step 3: Calculate amounts
    # If "med IVA/MVA inkludert/incluído" — amount is gross (including VAT)
    lower = prompt.lower()
    is_gross = any(w in lower for w in [
        "inkludert", "inkl", "included", "incluído", "incluido", "inclus",
        "einschließlich", "inklusive", "med iva", "com iva", "con iva",
        "avec tva", "mit mwst",
    ])
    if is_gross:
        gross_amount = amount
        net_amount = round(gross_amount / (1 + vat_pct / 100), 2)
        vat_amount = round(gross_amount - net_amount, 2)
    else:
        net_amount = amount
        vat_amount = round(net_amount * vat_pct / 100, 2)
        gross_amount = net_amount + vat_amount

    # Step 4: Find voucher type
    voucher_type_id = None
    vt_r = client.get("/ledger/voucherType", params={"fields": "id,name", "count": "20"})
    if "values" in vt_r and vt_r["values"]:
        for vt in vt_r["values"]:
            name_l = vt.get("name", "").lower()
            if any(w in name_l for w in ["leverandør", "innkjøp", "purchase", "supplier", "manuelt", "manual"]):
                voucher_type_id = vt["id"]
                break
        if not voucher_type_id:
            voucher_type_id = vt_r["values"][0]["id"]

    # Step 5: Create voucher with postings
    postings = [
        {"date": date, "account": {"id": expense_acct_id}, "amount": net_amount},  # Debit expense
    ]
    if vat_acct_id and vat_amount > 0:
        postings.append({"date": date, "account": {"id": vat_acct_id}, "amount": vat_amount})  # Debit input VAT
    postings.append({"date": date, "account": {"id": supplier_acct_id}, "amount": -gross_amount, "customer": {"id": supplier_id}})  # Credit supplier

    voucher_body: dict[str, Any] = {
        "date": date,
        "description": inv_ref or f"Faktura fra {company}",
        "postings": postings,
    }
    if voucher_type_id:
        voucher_body["type"] = {"id": voucher_type_id}

    result = client.post("/ledger/voucher", json_body=voucher_body)
    logger.info(f"Create supplier invoice voucher: {result}")
    if "value" in result:
        return f"OK: Supplier invoice from {company} registered (voucher ID: {result['value']['id']})"

    # Retry without customer on postings (some sandboxes don't support it)
    for p in postings:
        p.pop("customer", None)
    result2 = client.post("/ledger/voucher", json_body=voucher_body)
    if "value" in result2:
        return f"OK: Supplier invoice from {company} registered (voucher ID: {result2['value']['id']})"

    # Retry without voucher type
    if "type" in voucher_body:
        del voucher_body["type"]
        result3 = client.post("/ledger/voucher", json_body=voucher_body)
        if "value" in result3:
            return f"OK: Supplier invoice from {company} registered (voucher ID: {result3['value']['id']})"

    return f"ERR: {result}"


def _enable_module(client: TripletexClient, module_field: str) -> bool:
    """Helper: enable a company module. Returns True on success."""
    modules = client.get("/company/modules", params={"fields": "*"})
    if "value" not in modules:
        logger.warning(f"Could not GET modules: {modules}")
        return False
    mod_data = modules["value"]
    mod_data[module_field] = True
    for key in list(mod_data.keys()):
        if key in ("changes", "url"):
            del mod_data[key]
    result = client.put("/company/modules", json_body=mod_data)
    logger.info(f"Enable {module_field}: {result}")
    return "value" in result


def exec_create_dimension(prompt: str, client: TripletexClient) -> str:
    """Create custom dimension(s) with values, optionally create a linked voucher."""
    today = datetime.date.today().isoformat()
    date = extract_date(prompt) or today

    # Parse all quoted strings (single, double, guillemets, smart quotes)
    QUOTE_OPEN = r"""[\"\u201c\u00ab'\u2018]"""
    QUOTE_CLOSE = r"""[\"\u201d\u00bb'\u2019]"""
    QUOTE_ANY = r"""[\"\u201c\u201d\u00ab\u00bb'\u2018\u2019]"""
    all_quoted = re.findall(QUOTE_OPEN + r'([^\"\u201c\u201d\u00ab\u00bb\'\u2018\u2019]+)' + QUOTE_CLOSE, prompt)

    # Parse dimension name — first quoted string after "dimension" keyword
    dim_name = None
    dim_m = re.search(
        r'(?:dimension|dimensjon|dimensi\u00f3n|dimens\u00e3o|Dimension)\s+'
        r'(?:comptable\s+)?(?:personnalis\u00e9e?\s+)?(?:tilpasset\s+)?(?:personalizada?\s+)?'
        + QUOTE_OPEN + r'([^\"\u201c\u201d\u00ab\u00bb\'\u2018\u2019]+)' + QUOTE_CLOSE,
        prompt, re.IGNORECASE
    )
    if dim_m:
        dim_name = dim_m.group(1).strip()
    elif all_quoted:
        dim_name = all_quoted[0]

    if not dim_name:
        return "ERR: Could not extract dimension name"

    # Parse dimension values — quoted strings after "valeurs/verdier/values/Werte"
    dim_values: list[str] = []
    values_section = re.search(
        r'(?:valeurs?|verdier|verdiar|valores?|values?|Werte?)\s+'
        r'(.+?)(?:\.\s*(?:Puis|Then|Deretter|Dann|Luego|Depois|Og|And|Und|Et|Y|E)|$)',
        prompt, re.IGNORECASE | re.DOTALL
    )
    if values_section:
        dim_values = re.findall(
            QUOTE_OPEN + r'([^\"\u201c\u201d\u00ab\u00bb\'\u2018\u2019]+)' + QUOTE_CLOSE,
            values_section.group(1)
        )
    if not dim_values:
        dim_values = [q for q in all_quoted if q != dim_name]

    # Step 1: Enable custom dimension module
    _enable_module(client, "moduleCustomDimension")

    # Step 2: Create the dimension
    dim_r = client.post("/dimension", json_body={"name": dim_name})
    logger.info(f"Create dimension '{dim_name}': {dim_r}")
    if "value" not in dim_r:
        # Retry: maybe module needs different approach or dimension already exists
        dim_r = client.get("/dimension", params={"name": dim_name, "fields": "id,name", "count": "5"})
        if "values" in dim_r and dim_r["values"]:
            dim_r = {"value": dim_r["values"][0]}
        else:
            return f"ERR: Could not create dimension '{dim_name}': {dim_r}"
    dim_id = dim_r["value"]["id"]

    # Step 3: Create dimension values
    value_ids: dict[str, int] = {}
    for val_name in dim_values:
        val_r = client.post(f"/dimension/{dim_id}/dimensionValue", json_body={"name": val_name})
        logger.info(f"Create dim value '{val_name}': {val_r}")
        if "value" in val_r:
            value_ids[val_name] = val_r["value"]["id"]

    # Step 4: Check if prompt also asks for a voucher/journal entry
    lower = prompt.lower()
    has_voucher = any(w in lower for w in [
        'comptabilisez', 'bilag', 'voucher', 'journal entry', 'pi\u00e8ce',
        'bokf\u00f8r', 'buchung', 'asiento', 'lan\u00e7amento', 'comprobante',
        '\u00e9criture', 'ecriture', 'posteringsbilag', 'postering',
        'comptabiliser', 'verbuchen', 'contabilizar',
    ])
    if not has_voucher:
        return f"OK: Dimension '{dim_name}' created (ID: {dim_id}) with {len(value_ids)} values"

    # Step 5: Parse voucher info
    amount = extract_amount(prompt)
    account_m = re.search(r'(?:konto|account|compte|cuenta|conta|Konto)\s*:?\s*(\d{4})', prompt, re.IGNORECASE)
    account_num = account_m.group(1) if account_m else None

    if not account_num or not amount:
        return f"OK: Dimension '{dim_name}' created (ID: {dim_id}) with {len(value_ids)} values"

    # Find which dimension value to link
    linked_value_name = None
    link_m = re.search(
        r'(?:li\u00e9e?\s+\u00e0|linked?\s+to|knyttet\s+til|verkn\u00fcpft\s+mit|vinculad[oa]\s+[a\u00e0]|ligada?\s+[a\u00e0])',
        prompt, re.IGNORECASE
    )
    if link_m:
        rest = prompt[link_m.end():]
        val_m = re.search(
            QUOTE_OPEN + r'([^\"\u201c\u201d\u00ab\u00bb\'\u2018\u2019]+)' + QUOTE_CLOSE, rest
        )
        if val_m and val_m.group(1).strip() in value_ids:
            linked_value_name = val_m.group(1).strip()
    if not linked_value_name:
        for q in reversed(all_quoted):
            if q in value_ids:
                linked_value_name = q
                break

    # Step 6: Look up the debit account
    acct = client.get("/ledger/account", params={"number": account_num, "fields": "id,number", "count": "1"})
    if "values" not in acct or not acct["values"]:
        return f"ERR: Account {account_num} not found"
    acct_id = acct["values"][0]["id"]

    # Step 7: Find a contra account
    contra_id = None
    for contra_num in ["2400", "2920", "1920", "1900"]:
        contra = client.get("/ledger/account", params={"number": contra_num, "fields": "id", "count": "1"})
        if "values" in contra and contra["values"]:
            contra_id = contra["values"][0]["id"]
            break
    if not contra_id:
        return "ERR: No contra account found"

    # Step 8: Find a manual voucher type
    voucher_type_id = None
    vt_r = client.get("/ledger/voucherType", params={"fields": "id,name", "count": "20"})
    if "values" in vt_r and vt_r["values"]:
        for vt in vt_r["values"]:
            name_l = vt.get("name", "").lower()
            if any(w in name_l for w in ["manuelt", "manual", "diverse", "journal"]):
                voucher_type_id = vt["id"]
                break
        if not voucher_type_id:
            voucher_type_id = vt_r["values"][0]["id"]

    # Step 9: Build and create voucher
    debit_posting: dict[str, Any] = {
        "date": date,
        "account": {"id": acct_id},
        "amount": amount,
    }
    if linked_value_name and linked_value_name in value_ids:
        debit_posting["customDimensionValue1"] = {"id": value_ids[linked_value_name]}

    credit_posting: dict[str, Any] = {
        "date": date,
        "account": {"id": contra_id},
        "amount": -amount,
    }

    voucher_body: dict[str, Any] = {
        "date": date,
        "description": dim_name,
        "postings": [debit_posting, credit_posting],
    }
    if voucher_type_id:
        voucher_body["type"] = {"id": voucher_type_id}

    result = client.post("/ledger/voucher", json_body=voucher_body)
    logger.info(f"Create dimension voucher: {result}")
    if "value" in result:
        return f"OK: Dimension '{dim_name}' and voucher created"

    # Retry without type
    if voucher_type_id:
        del voucher_body["type"]
        result2 = client.post("/ledger/voucher", json_body=voucher_body)
        if "value" in result2:
            return f"OK: Dimension '{dim_name}' and voucher created"

    return f"ERR: Voucher failed: {result}"


# ─── LLM fallback (Vertex AI / Google AI) ──────────────────────────────────

# Import the comprehensive SYSTEM_PROMPT from prompts.py
from prompts import SYSTEM_PROMPT as _FULL_PROMPT

SYSTEM_INSTRUCTION = _FULL_PROMPT + """

## LLM RESPONSE FORMAT
Respond ONLY with JSON. No prose, no markdown.
Format: {"actions": [{"method":"GET|POST|PUT|DELETE","path":"/endpoint","body":{...},"params":{...}}, ...]}
After receiving results, respond with more actions or: {"done": true, "summary": "..."}

IMPORTANT: In the "path" field, do NOT include the /v2 prefix. Use "/customer" not "/v2/customer". The base URL already includes /v2.

## ABSOLUTELY FORBIDDEN (will cause 422 errors)
1. NEVER use "ADMINISTRATOR" as userType. ONLY "STANDARD" (with email) or "NO_ACCESS" (no email needed).
2. NEVER use numeric userType values (0, 1, 2, etc.). userType MUST be a string.
3. NEVER include departureDate or returnDate on travelExpense — these fields DO NOT EXIST.
   Travel expenses only have: employee(id), title, date, project(id), department(id).
4. NEVER set vatType on order lines — omit it entirely.
5. NEVER send a request body for action endpoints (/:invoice, /:payment) — use query params only.

CRITICAL for action endpoints (/:invoice, /:payment, /:createCreditNote, /:deliver, /:approve):
- Use "method": "PUT"
- For /:invoice and /:payment, put parameters in "params" (query params), NOT "body"
- Example payment: {"method":"PUT","path":"/invoice/123/:payment","params":{"paymentDate":"2025-01-20","paymentTypeId":"1","paidAmount":"5000","paidAmountCurrency":"5000"}}
- Example invoice: {"method":"PUT","path":"/order/123/:invoice","params":{"invoiceDate":"2025-01-20","sendToCustomer":"false"}}

For POST /order/orderline/list, body must be a raw JSON array:
- {"method":"POST","path":"/order/orderline/list","body":[{"order":{"id":N},"description":"...","count":1,"unitPriceExcludingVatCurrency":100}]}

NEVER set vatType on order lines — omit it entirely.

## TRAVEL EXPENSE WORKFLOW (MUST FOLLOW EXACTLY)
1. Find employee: GET /employee?email=... or ?firstName=...&lastName=...
2. If no employee found, create: POST /employee {firstName, lastName, userType: "NO_ACCESS", department: {id: N}}
3. Create travel expense: POST /travelExpense {employee: {id: N}, title: "..."}
   ONLY these fields exist: employee, title, date, project, department. NO departureDate, NO returnDate.
4. Look up cost categories: GET /travelExpense/costCategory
5. Look up payment types: GET /travelExpense/paymentType
6. Look up VAT types: GET /ledger/vatType (pick 0% for expenses)
7. Add cost lines: POST /travelExpense/cost {travelExpense:{id}, costCategory:{id}, vatType:{id}, currency:{id:1}, paymentType:"EMPLOYEE", date:"YYYY-MM-DD", count:N, rate:AMOUNT, amount:TOTAL}
   For per diem/diett: count = number of days, rate = daily rate, amount = days × rate
   For expenses (flight, taxi, hotel): count = 1, rate = amount, amount = amount

## EMPLOYEE CREATION (MUST FOLLOW EXACTLY)
1. GET /department?fields=id&count=1 — find a department
2. If no department, POST /department {name: "Hovedavdeling", departmentNumber: "1"}
3. POST /employee {firstName, lastName, userType: "STANDARD", email: "...", department: {id: N}}
   - If no email provided: use userType: "NO_ACCESS" (no email needed)
   - NEVER use "ADMINISTRATOR" — it does not exist as a valid userType
"""


def run_llm_agent(prompt: str, files: list[dict], client: TripletexClient) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "ERR: google-genai not installed"

    if api_key:
        gc = genai.Client(api_key=api_key)
    else:
        project = os.environ.get("GCP_PROJECT", "ai-nm26osl-1724")
        location = os.environ.get("GCP_LOCATION", "us-central1")
        gc = genai.Client(vertexai=True, project=project, location=location)

    # Build initial message parts — include file attachments for vision
    user_parts = [types.Part.from_text(text=f"Task:\n{prompt}\n\nRespond with JSON actions.")]

    for f in files:
        try:
            file_data = base64.b64decode(f.get("content_base64", ""))
            mime = f.get("mime_type", "application/octet-stream")
            fname = f.get("filename", "attachment")
            if mime.startswith("image/") or mime == "application/pdf":
                user_parts.append(types.Part.from_text(text=f"\n[Attached file: {fname}]"))
                user_parts.append(types.Part.from_bytes(data=file_data, mime_type=mime))
            else:
                # For other files, try to include as text
                try:
                    text_content = file_data.decode("utf-8")
                    user_parts.append(types.Part.from_text(text=f"\n[File: {fname}]\n{text_content[:5000]}"))
                except Exception:
                    user_parts.append(types.Part.from_text(text=f"\n[File: {fname} ({mime}) — binary, cannot display]"))
        except Exception as e:
            logger.warning(f"Failed to process file attachment: {e}")

    messages = [types.Content(role="user", parts=user_parts)]

    for iteration in range(25):
        resp = gc.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,
            ),
        )
        text = resp.text.strip()
        logger.info(f"LLM iter {iteration}: {text[:500]}")

        try:
            json_m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if json_m:
                text = json_m.group(1).strip()
            # Also try to find JSON object in prose
            if not text.startswith('{') and not text.startswith('['):
                json_m2 = re.search(r'(\{[\s\S]*\})', text)
                if json_m2:
                    text = json_m2.group(1).strip()
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Bad JSON from LLM: {text[:200]}")
            break

        if data.get("done"):
            return data.get("summary", "OK via LLM")

        results = []
        for a in data.get("actions", []):
            method = a.get("method", "GET").upper()
            path = a.get("path", "")
            params = a.get("params")
            body = a.get("body")
            if params:
                params = {k: str(v) for k, v in params.items()}
            if method == "GET":
                r = client.get(path, params=params)
            elif method == "POST":
                r = client.post(path, json_body=body, params=params)
            elif method == "PUT":
                r = client.put(path, json_body=body, params=params)
            elif method == "DELETE":
                r = client.delete(path)
            else:
                r = {"error": f"Unknown method: {method}"}
            results.append({"action": f"{method} {path}", "result": r})

        messages.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
        messages.append(types.Content(role="user", parts=[types.Part.from_text(
            text=f"Results:\n{json.dumps(results, default=str)[:12000]}\n\nContinue with next actions or {{\"done\": true, \"summary\": \"...\"}}."
        )]))

    return f"OK: LLM done. Calls: {len(client.call_log)}"


# ─── Executor map ───────────────────────────────────────────────────────────

EXECUTORS = {
    "create_employee": exec_create_employee,
    "create_customer": exec_create_customer,
    "create_product": exec_create_product,
    "create_department": exec_create_department,
    "create_project": exec_create_project,
    "create_invoice": exec_create_invoice,
    "create_supplier": exec_create_supplier,
    "delete_travel_expense": exec_delete_travel_expense,
    "create_travel_expense": exec_create_travel_expense,
    "register_payment": exec_register_payment,
    "update_employee": exec_update_employee,
    "update_customer": exec_update_customer,
    "create_contact": exec_create_contact,
    "create_credit_note": exec_create_credit_note,
    "delete_voucher": exec_delete_voucher,
    "send_invoice": exec_send_invoice,
    "approve_travel_expense": exec_approve_travel_expense,
    "deliver_travel_expense": exec_deliver_travel_expense,
    "create_timesheet": exec_create_timesheet,
    "create_voucher": exec_create_voucher,
    "enable_module": exec_enable_module,
    "create_dimension": exec_create_dimension,
    "create_supplier_invoice": exec_create_supplier_invoice,
}


def run_agent(prompt: str, files: list[dict], client: TripletexClient, **kwargs) -> str:
    """Main entry point. Regex-based first, LLM fallback for complex/unknown tasks."""
    logger.info(f"Prompt: {prompt[:300]}")

    task_type = detect_task_type(prompt)
    logger.info(f"Detected: {task_type}")

    if task_type and task_type in EXECUTORS:
        try:
            result = EXECUTORS[task_type](prompt, client)
            logger.info(f"Result: {result}")
            if not result.startswith("ERR:"):
                return result
            logger.warning(f"Regex failed: {result}, trying LLM")
        except Exception as e:
            logger.error(f"Executor error: {e}")

    # LLM fallback
    logger.info("Falling back to LLM")
    try:
        return run_llm_agent(prompt, files, client)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"ERR: {e}"
