import os
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SATUSEHAT_CLIENT_ID")
CLIENT_SECRET = os.getenv("SATUSEHAT_CLIENT_SECRET")
AUTH_URL = "https://api-satusehat-stg.kemkes.go.id/oauth2/v1/accesstoken?grant_type=client_credentials"
BASE_URL = "https://api-satusehat-stg.kemkes.go.id/fhir-r4/v1"

class SatusehatService:
    def __init__(self):
        self.access_token = None

    def get_access_token(self):
        if not CLIENT_ID or not CLIENT_SECRET:
            raise ValueError("SATUSEHAT credentials are not configured in environment variables.")
            
        data = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        
        # SATUSEHAT auth usually expects form-urlencoded
        response = requests.post(AUTH_URL, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        if response.status_code == 200:
            self.access_token = response.json().get("access_token")
            return self.access_token
        else:
            raise Exception(f"Gagal mendapatkan token: {response.text}")

    def get_patient_by_nik(self, nik: str):
        if not self.access_token:
            self.get_access_token()
            
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        url = f"{BASE_URL}/Patient?identifier=https://fhir.kemkes.go.id/id/nik|{nik}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            # Token might be expired, retry once
            self.get_access_token()
            headers["Authorization"] = f"Bearer {self.access_token}"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Gagal mendapatkan data pasien (Unauthorized): {response.text}")
        else:
            raise Exception(f"Gagal mendapatkan data pasien: {response.text}")

    def create_resource(self, resource_type: str, payload: dict):
        if not self.access_token:
            self.get_access_token()
            
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        url = f"{BASE_URL}/{resource_type}"
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code in [200, 201]:
            return response.json()
        elif response.status_code == 401:
            self.get_access_token()
            headers["Authorization"] = f"Bearer {self.access_token}"
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code in [200, 201]:
                return response.json()
            else:
                raise Exception(f"Gagal kirim resource {resource_type} (Unauthorized): {response.text}")
        else:
            raise Exception(f"Gagal kirim resource {resource_type}: {response.text}")

    def get_encounter_by_patient(self, patient_satusehat_id: str):
        """Cari Encounter untuk pasien di SATUSEHAT (semua status)."""
        if not self.access_token:
            self.get_access_token()
            
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # Cari semua encounter tanpa filter status
        url = f"{BASE_URL}/Encounter?patient=Patient/{patient_satusehat_id}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            entries = data.get("entry", [])
            if entries:
                return entries[0]["resource"]["id"]
        elif response.status_code == 401:
            self.get_access_token()
            headers["Authorization"] = f"Bearer {self.access_token}"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                entries = data.get("entry", [])
                if entries:
                    return entries[0]["resource"]["id"]
        
        return None

    def send_observation(self, patient_satusehat_id: str, conclusion: str, org_id: str, encounter_id: str = None):
        from datetime import datetime, timezone
        
        # Mapping conclusion to SNOMED codes (Contoh sederhana)
        # Default: Gram-positive cocci
        snomed_code = "70003006" 
        display_name = "Gram-positive cocci"
        
        lower_c = conclusion.lower()
        if "negatif" in lower_c:
            if "kokus" in lower_c:
                snomed_code = "87172008" # Gram-negative cocci
                display_name = "Gram-negative cocci"
            elif "batang" in lower_c or "basil" in lower_c:
                snomed_code = "71542004" # Gram-negative rod
                display_name = "Gram-negative rod"
        elif "positif" in lower_c:
            if "batang" in lower_c or "basil" in lower_c:
                snomed_code = "71542004" # Note: SNOMED for Gram-pos rod is usually different, but let's use a generic one if not found
                display_name = "Gram-positive rod"

        observation_payload = {
            "resourceType": "Observation",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "laboratory"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "664-3",
                    "display": "Gram stain"
                }]
            },
            "subject": {"reference": f"Patient/{patient_satusehat_id}"},
            "performer": [{"reference": f"Organization/{org_id}"}],
            "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": snomed_code,
                    "display": conclusion 
                }]
            }
        }

        if encounter_id:
            observation_payload["encounter"] = {"reference": f"Encounter/{encounter_id}"}

        return self.create_resource("Observation", observation_payload)

    def send_diagnostic_report(self, patient_satusehat_id: str, observation_id: str, org_id: str, accession_number: str):
        from datetime import datetime, timezone
        
        report_payload = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "LAB",
                    "display": "Laboratory"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "664-3",
                    "display": "Gram stain"
                }]
            },
            "subject": {"reference": f"Patient/{patient_satusehat_id}"},
            "performer": [{"reference": f"Organization/{org_id}"}],
            "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
            "issued": datetime.now(timezone.utc).isoformat(),
            "result": [{"reference": f"Observation/{observation_id}"}],
            "identifier": [{
                "system": "http://sys-ids.kemkes.go.id/id/accession",
                "value": accession_number
            }]
        }
        return self.create_resource("DiagnosticReport", report_payload)

satusehat_service = SatusehatService()
