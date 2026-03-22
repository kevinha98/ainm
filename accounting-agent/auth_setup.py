"""Create Application Default Credentials via browser-based OAuth2 flow.
Uses the same approach as `gcloud auth application-default login`."""

import json
import os
from google_auth_oauthlib.flow import InstalledAppFlow

# Standard gcloud desktop app OAuth2 credentials (public, in gcloud source code)
CLIENT_CONFIG = {
    "installed": {
        "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
        "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }
}

SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language",
]

def main():
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, scopes=SCOPES)
    creds = flow.run_local_server(port=0)

    # Save as application_default_credentials.json
    adc_dir = os.path.join(os.environ["APPDATA"], "gcloud")
    os.makedirs(adc_dir, exist_ok=True)
    adc_path = os.path.join(adc_dir, "application_default_credentials.json")

    adc_data = {
        "client_id": CLIENT_CONFIG["installed"]["client_id"],
        "client_secret": CLIENT_CONFIG["installed"]["client_secret"],
        "refresh_token": creds.refresh_token,
        "type": "authorized_user",
    }

    with open(adc_path, "w") as f:
        json.dump(adc_data, f, indent=2)

    print(f"Credentials saved to: {adc_path}")
    print("You can now use Vertex AI / Google Cloud APIs locally.")

if __name__ == "__main__":
    main()
