import os
import ssl
import sys
import tempfile
import threading
import logging
from datetime import datetime
from flask import Flask, request
import requests
from google import genai
from google.genai import types

# הגדרת logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# מניעת ריצה כפולה של תמלול
transcription_lock = threading.Lock()
is_transcribing = False

# הגדרת UTF-8 ל-Windows
if sys.platform == 'win32':
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'he_IL.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'Hebrew_Israel.1255')
        except:
            pass

# Monkey patch ל-SSL
_original_create_default_context = ssl.create_default_context

def _create_unverified_context(*args, **kwargs):
    context = _original_create_default_context(*args, **kwargs)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

ssl.create_default_context = _create_unverified_context

app = Flask(__name__)

# הגדרות
YMOT_TOKEN = os.getenv('YMOT_TOKEN', 'YOUR_TOKEN_HERE')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_KEY_HERE')

class YemotTranscriptionService:
    """
    שירות תמלול קבצים מימות המשיח
    """
    def __init__(self, yemot_token: str, gemini_api_key: str):
        self.yemot_token = yemot_token
        self.gemini_api_key = gemini_api_key
        self.base_url = "https://www.call2all.co.il/ym/api/"
        self.client = genai.Client(api_key=gemini_api_key)
    
    def download_file_from_path(self, file_path: str):
        """
        מוריד קובץ מה-API של ימות מנתיב מלא
        """
        url = f"{self.base_url}DownloadFile"
        params = {
            'token': self.yemot_token,
            'path': file_path
        }

        try:
            logger.debug(f"Downloading file from path: {file_path}")
            response = requests.post(url, data=params, verify=False)
            logger.debug(f"Download response status: {response.status_code}")
            if response.status_code == 200:
                logger.debug(f"Downloaded {len(response.content)} bytes")
                return response.content
            else:
                logger.debug(f"Download failed with status {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"שגיאה בהורדת קובץ: {e}")
            return None
    
    def upload_tts_file(self, extension: str, file_name: str, content: str):
        """
        מעלה קובץ TTS לשלוחה מסוימת
        """
        url = f"{self.base_url}UploadTextFile"
        params = {
            'token': self.yemot_token,
            'path': f'ivr2:{extension}/{file_name}',
            'contents': content,
            'convertAudio': '1'
        }
        
        try:
            logger.debug(f"Uploading TTS file: {file_name} to extension {extension}")
            logger.debug(f"Content length: {len(content)} characters")
            response = requests.post(url, data=params, verify=False)
            logger.debug(f"Upload response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Upload response: {data}")
                return data.get('responseStatus') == 'OK'
            else:
                logger.debug(f"Upload failed with status {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"שגיאה בהעלאת קובץ: {e}")
            return False
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """
        מתמלל קובץ אודיו ב-Gemini עם תמיכה בארמית, עברית ולשון הקודש
        """
        try:
            logger.debug(f"Starting transcription of: {audio_file_path}")
            # העלאת הקובץ ל-Gemini עם mime_type מפורש ב-config
            with open(audio_file_path, 'rb') as f:
                uploaded_file = self.client.files.upload(file=f, config={'mime_type': 'audio/wav'})
            logger.debug(f"File uploaded to Gemini: {uploaded_file.name}")
            
            # תמלול הקובץ
            response = self.client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=[
                    "Transcribe the following audio file accurately. The audio contains speech in Aramaic, Hebrew, and/or Biblical Hebrew - possibly mixed together in the same recording. The speech may be pronounced with Ashkenazi or Hasidic Jewish pronunciation. Please transcribe exactly what is said, preserving the original language, words, and pronunciation. Do not translate or summarize. Return ONLY the transcription text without any explanations, notes, or additional content.",
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type='audio/wav'
                    )
                ]
            )
            
            # חילוץ התמלול מהתגובה
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response has text attribute: {hasattr(response, 'text')}")
            if hasattr(response, 'text'):
                if response.text:
                    transcription = response.text.strip()
                    logger.debug(f"Transcription completed. Length: {len(transcription)} characters")
                    return transcription
                else:
                    logger.debug("Response.text is empty, trying candidates")
            if hasattr(response, 'candidates') and response.candidates:
                logger.debug(f"Found {len(response.candidates)} candidates")
                # נסה לקחת מ-candidates
                for i, candidate in enumerate(response.candidates):
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for j, part in enumerate(candidate.content.parts):
                            if hasattr(part, 'text') and part.text:
                                transcription = part.text.strip()
                                logger.debug(f"Transcription from candidates. Length: {len(transcription)} characters")
                                return transcription
            logger.error(f"Response exists but no text found")
            return None
            
        except Exception as e:
            logger.error(f"שגיאה בתמלול: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # מחיקת הקובץ המועלה מ-Gemini
            try:
                if 'uploaded_file' in locals():
                    self.client.files.delete(name=uploaded_file.name)
                    logger.debug("Deleted file from Gemini")
            except:
                pass
    
    def process_transcription_from_path(self, file_path: str, target_extension: str) -> str:
        """
        מעבדת תמלול מנתיב קובץ מלא
        """
        logger.debug(f"Processing transcription from path: {file_path}")

        # הורדת הקובץ
        audio_data = self.download_file_from_path(file_path)

        if not audio_data:
            return "id_list_message=download_failed"

        # שמירת הקובץ באופן זמני
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        try:
            # תמלול הקובץ
            transcription = self.transcribe_audio(temp_file_path)

            if not transcription:
                return "id_list_message=transcription_failed"

            # יצירת שם קובץ עם timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tts_filename = f"transcription_{timestamp}.tts"

            # העלאת התמלול כקובץ TTS לשלוחת היעד
            upload_success = self.upload_tts_file(target_extension, tts_filename, transcription)

            if not upload_success:
                return "id_list_message=upload_failed"

            # הצלחה
            return "id_list_message=success"

        finally:
            # מחיקת הקובץ הזמני
            try:
                os.unlink(temp_file_path)
            except:
                pass

# משתנה גלובלי לשירות
service = YemotTranscriptionService(YMOT_TOKEN, GEMINI_API_KEY)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    נקודת קצה לתמלול - מקבלת נתיב קובץ ומעלה את התמלול לשלוחה 8
    """
    global is_transcribing
    try:
        logger.debug("Transcribe endpoint called")

        # קבלת הנתיב מהפרמטר file
        file_path = request.form.get('file')

        if not file_path:
            logger.debug("No file path provided")
            return "id_list_message=no_file_path"

        logger.debug(f"File path received: {file_path}")

        # הוספת ivr2: לנתיב
        full_path = f"ivr2:{file_path}"
        logger.debug(f"Full path: {full_path}")

        # בדיקה אם כבר יש תמלול פעיל
        with transcription_lock:
            if is_transcribing:
                logger.debug("Transcription already in progress, ignoring duplicate request")
                return "id_list_message=already_transcribing"
            is_transcribing = True

        # תמלול ברקע
        def process_in_background():
            global is_transcribing
            logger.debug("Starting background transcription process")
            try:
                result = service.process_transcription_from_path(full_path, '8')
                logger.debug(f"Background transcription result: {result}")
            finally:
                with transcription_lock:
                    is_transcribing = False

        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()

        return "id_list_message=file_sent_for_transcription"

    except Exception as e:
        logger.error(f"שגיאה ב-transcribe: {e}")
        import traceback
        traceback.print_exc()
        with transcription_lock:
            is_transcribing = False
        return "id_list_message=error"

@app.route('/health', methods=['GET'])
def health():
    """
    בדיקת בריאות
    """
    return "OK"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
