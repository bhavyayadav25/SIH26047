# MediKiosk AI-5H — Voice & Accessibility

AI-5H adds a patient-facing accessibility layer to the cumulative SIH26047 backend. It complements the existing voice transcription/TTS stack with persisted per-patient accessibility preferences and explicit capability discovery.

## Endpoints

- `GET /api/accessibility/capabilities`
- `GET /api/patients/{patient_id}/accessibility`
- `PUT /api/patients/{patient_id}/accessibility`
- `GET /api/voice/status`
- Existing `POST /api/voice/transcribe`
- Existing `POST /api/voice/speak`

## Accessibility controls

- Language selection across the existing supported Indian-language UI set
- Touch, voice, or hybrid input mode
- Font scaling
- High contrast
- Reduced motion
- Captions
- Audio enable/disable
- Audio speed
- Assisted mode

Voice remains optional: touch/text fallback is available even when local speech packages/models are not installed. Server TTS languages are reported explicitly rather than pretending every UI language has a configured server voice.

## Privacy and safety

- Accessibility preferences contain no clinical data.
- Patients may modify only their own preferences.
- Clinical/administrative roles may read preferences for operational support.
- Voice transcription rejects patient attempts to submit audio for another patient account.
- Raw patient audio is not persisted by the existing voice transcription implementation; only transcript metadata is stored after successful transcription.
- AI-5H does not diagnose, prescribe, or make clinical decisions.
