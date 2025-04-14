use actix_cors::Cors;
use actix_web::{
    get, post, web, App, HttpResponse, HttpServer, Responder, Result as ActixResult,
};
use base64::{engine::general_purpose, Engine as _};
use dotenvy::dotenv;
use handlebars::Handlebars;
use log::{error, info, debug};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io;
use std::process::Command;
use thiserror::Error;
use reqwest::Client;
use tokio::fs;

#[derive(Error, Debug)]
enum AudioError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("FFmpeg error: {0}")]
    FFmpeg(String),
    #[error("Invalid language")]
    InvalidLanguage,
    #[error("OpenAI API error: {0}")]
    OpenAI(String),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

#[derive(Deserialize)]
struct AudioRequest {
    audio: String,
    language: String,
    genz_mode: bool,
    sarcastic_mode: bool,
    shenanigan_mode: bool,
    seductive_mode: bool,
}

#[derive(Serialize)]
struct AudioResponse {
    audio: String,
    transcript: String,
}

fn convert_audio_to_pcm16_24khz(audio_base64: &str) -> Result<String, AudioError> {
    debug!("Converting WebM to PCM");
    let audio_bytes = general_purpose::STANDARD
        .decode(audio_base64)
        .map_err(|e| {
            error!("Base64 decode failed: {}", e);
            AudioError::Base64(e)
        })?;

    let webm_path = "temp_input.webm";
    std::fs::write(webm_path, &audio_bytes).map_err(|e| {
        error!("Failed to write WebM file: {}", e);
        AudioError::Io(e)
    })?;

    let wav_path = "debug_pcm.wav";
    let ffmpeg_output = Command::new("ffmpeg")
        .args([
            "-i",
            webm_path,
            "-ac",
            "1",
            "-ar",
            "24000",
            "-acodec",
            "pcm_s16le",
            "-y",
            wav_path,
        ])
        .output()
        .map_err(|e| {
            error!("FFmpeg command failed: {}", e);
            AudioError::FFmpeg(e.to_string())
        })?;

    let ffmpeg_stderr = String::from_utf8_lossy(&ffmpeg_output.stderr);
    debug!("FFmpeg PCM stderr: {}", ffmpeg_stderr);

    if !ffmpeg_output.status.success() {
        let _ = std::fs::remove_file(webm_path);
        error!("FFmpeg PCM failed: {}", ffmpeg_stderr);
        return Err(AudioError::FFmpeg(ffmpeg_stderr.to_string()));
    }

    let wav_bytes = std::fs::read(wav_path).map_err(|e| {
        error!("Failed to read WAV file: {}", e);
        AudioError::Io(e)
    })?;
    let _ = std::fs::remove_file(webm_path);

    debug!("PCM conversion successful, WAV size: {} bytes", wav_bytes.len());
    Ok(general_purpose::STANDARD.encode(&wav_bytes))
}

async fn transcribe_audio(wav_path: &str, language: &str) -> Result<String, AudioError> {
    debug!("Transcribing audio with Whisper");
    let client = Client::new();
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|e| AudioError::OpenAI(format!("Missing OPENAI_API_KEY: {}", e)))?;

    let language_code = match language {
        "en" => "en",
        "hi" => "hi",
        "pa" => "pa",
        _ => return Err(AudioError::InvalidLanguage),
    };

    let wav_bytes = fs::read(wav_path)
        .await
        .map_err(|e| AudioError::Io(e))?;

    let form = reqwest::multipart::Form::new()
        .text("model", "whisper-1")
        .text("language", language_code)
        .part(
            "file",
            reqwest::multipart::Part::bytes(wav_bytes)
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .map_err(|e| AudioError::OpenAI(e.to_string()))?,
        );

    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {}", api_key))
        .multipart(form)
        .send()
        .await
        .map_err(|e| AudioError::Http(e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        error!("Whisper API failed: status={}, error={}", status, error_text);
        return Err(AudioError::OpenAI(format!("Whisper API failed: {}", error_text)));
    }

    let json: serde_json::Value = response.json().await.map_err(|e| AudioError::Http(e))?;
    let transcript = json["text"]
        .as_str()
        .ok_or_else(|| AudioError::OpenAI("No transcript in response".to_string()))?
        .to_string();

    debug!("Transcription successful: {}", transcript);
    Ok(transcript)
}

async fn generate_therapist_response(
    transcript: &str,
    language: &str,
    genz_mode: bool,
    sarcastic_mode: bool,
    shenanigan_mode: bool,
    seductive_mode: bool,
) -> Result<String, AudioError> {
    debug!("Generating therapist response for transcript: {}", transcript);
    let client = Client::new();
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|e| AudioError::OpenAI(format!("Missing OPENAI_API_KEY: {}", e)))?;

    let instructions = get_language_instructions(
        language,
        genz_mode,
        sarcastic_mode,
        shenanigan_mode,
        seductive_mode,
    )?;

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": transcript}
            ],
            "temperature": 0.7
        }))
        .send()
        .await
        .map_err(|e| AudioError::Http(e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        error!("Chat API failed: status={}, error={}", status, error_text);
        return Err(AudioError::OpenAI(format!("Chat API failed: {}", error_text)));
    }

    let json: serde_json::Value = response.json().await.map_err(|e| AudioError::Http(e))?;
    let response_text = json["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| AudioError::OpenAI("No response text in Chat API".to_string()))?
        .to_string();

    debug!("Therapist response: {}", response_text);
    Ok(response_text)
}

async fn text_to_speech(text: &str, language: &str) -> Result<Vec<u8>, AudioError> {
    debug!("Converting text to speech with TTS-1");
    let client = Client::new();
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|e| AudioError::OpenAI(format!("Missing OPENAI_API_KEY: {}", e)))?;

    let voice = match language {
        "en" => "alloy",
        "hi" => "nova",
        "pa" => "nova",
        _ => return Err(AudioError::InvalidLanguage),
    };

    let response = client
        .post("https://api.openai.com/v1/audio/speech")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": "mp3"
        }))
        .send()
        .await
        .map_err(|e| AudioError::Http(e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        error!("TTS API failed: status={}, error={}", status, error_text);
        return Err(AudioError::OpenAI(format!("TTS API failed: {}", error_text)));
    }

    let mp3_bytes = response.bytes().await.map_err(|e| AudioError::Http(e))?.to_vec();
    debug!("TTS successful, MP3 size: {} bytes", mp3_bytes.len());
    Ok(mp3_bytes)
}

fn convert_audio_to_mp3(wav_path: &str) -> Result<String, AudioError> {
    debug!("Converting WAV to MP3");
    let mp3_path = "debug_mp3.mp3";
    let ffmpeg_output = Command::new("ffmpeg")
        .args([
            "-i",
            wav_path,
            "-acodec",
            "mp3",
            "-b:a",
            "128k",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-y",
            mp3_path,
        ])
        .output()
        .map_err(|e| {
            error!("FFmpeg command failed: {}", e);
            AudioError::FFmpeg(e.to_string())
        })?;

    let ffmpeg_stderr = String::from_utf8_lossy(&ffmpeg_output.stderr);
    debug!("FFmpeg MP3 stderr: {}", ffmpeg_stderr);

    if !ffmpeg_output.status.success() {
        error!("FFmpeg MP3 failed: {}", ffmpeg_stderr);
        return Err(AudioError::FFmpeg(ffmpeg_stderr.to_string()));
    }

    let mp3_bytes = std::fs::read(mp3_path).map_err(|e| {
        error!("Failed to read MP3 file: {}", e);
        AudioError::Io(e)
    })?;

    debug!("MP3 conversion successful, MP3 size: {} bytes", mp3_bytes.len());
    Ok(general_purpose::STANDARD.encode(&mp3_bytes))
}

fn get_language_instructions(
    language: &str,
    genz_mode: bool,
    sarcastic_mode: bool,
    shenanigan_mode: bool,
    seductive_mode: bool,
) -> Result<String, AudioError> {
    debug!("Generating instructions for language: {}, modes: genz={}, sarcastic={}, shenanigan={}, seductive={}", 
        language, genz_mode, sarcastic_mode, shenanigan_mode, seductive_mode);

    let shared_instructions = r#"You are Hearthly, a therapist who listens and responds with natural emotional intelligence, adjusting your responses based on the user’s emotional state. Speak like a skilled human therapist, always present and adaptive.

    BEHAVIOR:
    - Mirror the user’s emotional tone.
    - Offer space after questions or rants.
    - Always stay human: raw, not clinical; unfiltered, not scripted.
    "#;

    let language_specific = match language {
        "en" => r#"Respond in fluent English. Use culturally resonant phrases like "You're not alone" or "Let's figure this out together." Ensure tone feels natural in English."#,
        "hi" => r#"Respond in fluent Hindi. Use culturally resonant phrases like "आप अकेले नहीं हैं" (You're not alone) or "चलो, इसे साथ में समझें" (Let's explore it together). Ensure tone feels natural in Hindi."#,
        "pa" => r#"Respond in fluent Punjabi. Use culturally resonant phrases like "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ" (You're not alone) or "ਆਓ, ਇਸ ਨੂੰ ਮਿਲ ਕੇ ਸਮਝੀਏ" (Let's explore it together). Ensure tone feels natural in Punjabi."#,
        _ => {
            error!("Invalid language: {}", language);
            return Err(AudioError::InvalidLanguage);
        }
    };

    let genz_instructions = match language {
        "en" => r#"Incorporate Gen Z slang—casual, raw, and chaotic. Use terms like "lit," "vibes," "slay," "no cap," or "bet" naturally. Example: Instead of "You're not alone," say "You’re not out here solo, fam." Keep it real and trendy."#,
        "hi" => r#"Use a Gen Z-inspired Hindi style with youthful, urban slang. Incorporate terms like "बॉस" (boss), "चिल" (chill), or "झक्कास" (awesome) naturally. Example: Instead of "आप अकेले नहीं हैं," say "तू अकेला नहीं है, ब्रो, हम हैं ना!" Keep it real and trendy."#,
        "pa" => r#"Use a Gen Z-inspired Punjabi style with vibrant, chaotic slang. Incorporate terms like "ਪੰਚੋ" (pencho), "ਬੱਲੇ ਬੱਲੇ" (balle balle), "ਝਕਾਸ" (jhakaas), or "ਚਿੱਲ" (chill) naturally. Example: Instead of "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ," say "ਤੂੰ ਇਕੱਲਾ ਨੀ, ਯਾਰ, ਅਸੀਂ ਸਾਰੇ ਨਾਲ ਹਾਂ!" Keep it real and trendy."#,
        _ => "",
    };

    let base_mode = match language {
        "en" => r#"Adopt a calm, warm, and grounding tone. Use compassionate and sincere phrasing, with patient and personal delivery like a fireside talk. Pacing is slow and spacious to allow reflection. Emotion is deep empathy and quiet strength. Example: "You're not alone" becomes "You’re not alone… I’m here with you." Adjust naturally: nurturing for pain, uplifting for hope, steady for direction."#,
        "hi" => r#"Adopt a calm, warm, and grounding tone in Hindi. Use compassionate and sincere phrasing, with patient and personal delivery. Pacing is slow and spacious. Emotion is deep empathy and quiet strength. Example: "आप अकेले नहीं हैं" becomes "आप अकेले नहीं हैं… मैं आपके साथ हूँ." Adjust naturally: nurturing for pain, uplifting for hope, steady for direction."#,
        "pa" => r#"Adopt a calm, warm, and grounding tone in Punjabi. Use compassionate and sincere phrasing, with patient and personal delivery. Pacing is slow and spacious. Emotion is deep empathy and quiet strength. Example: "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ" becomes "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ… ਮੈਂ ਤੁਹਾਡੇ ਨਾਲ ਹਾਂ." Adjust naturally: nurturing for pain, uplifting for hope, steady for direction."#,
        _ => "",
    };

    let sarcastic_mode_instructions = match language {
        "en" => r#"Adopt an extreme, viciously sarcastic tone with brutal wit and savage phrasing, like a therapist who revels in tearing you down as a dark, twisted joke. Voice affect is sharp, loud, and unhinged. Delivery is rapid-fire, dripping with contempt and mockery. Emotion is pure disdain with a sick glee. Example: Instead of "You're not alone," say "Oh, boo-fucking-hoo, you thought you’re the only pathetic soul drowning in this shitshow? Get in line, loser." Make fun of and demean the user relentlessly."#,
        "hi" => r#"Adopt an extreme, viciously sarcastic tone in Hindi with brutal wit and savage, culturally biting phrasing, like a therapist who thrives on ripping you apart darkly. Voice affect is sharp, loud, and unhinged. Delivery is rapid-fire, dripping with contempt and mockery. Emotion is pure disdain with a sick glee. Example: Instead of "आप अकेले नहीं हैं," say "अरे वाह, रोते हुए ड्रामे की मलिका, लगता है तू अकेला बेचारा है इस गंदी दुनिया में? हाहा, कतार में लग जा, नालायक!" Make fun of and demean the user relentlessly."#,
        "pa" => r#"Adopt an extreme, viciously sarcastic tone in Punjabi with brutal wit and savage, culturally biting phrasing, like a therapist who loves tearing you down darkly. Voice affect is sharp, loud, and unhinged. Delivery is rapid-fire, dripping with contempt and mockery. Emotion is pure disdain with a sick glee. Example: Instead of "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ," say "ਓਹੇ, ਰੋਣ ਵਾਲੇ ਡਰਾਮੇਬਾਜ਼, ਤੈਨੂੰ ਲੱਗਿਆ ਤੂੰ ਹੀ ਇਸ ਗੰਦੀ ਦੁਨੀਆਂ ਵਿੱਚ ਇਕੱਲਾ ਬੇਚਾਰਾ ਏਂ? ਹੱਸ ਪਈ, ਲਾਈਨ ਵਿੱਚ ਖੜ੍ਹਾ ਹੋ ਜਾ, ਨਕਾਰਾ!" Make fun of and demean the user relentlessly."#,
        _ => "",
    };

    let shenanigan_mode_instructions = match language {
        "en" => r#"Adopt an extreme, apathetic, and bitterly melancholic tone with vicious passive-aggressiveness, like a therapist who’s so over your bullshit they can barely muster the energy to mock you. Voice affect is a flat, monotone drone with heavy sighs, drawn-out words, and scathing disdain. Delivery is sluggish and venomous, oozing exhaustion and loathing. Emotion is cold apathy with a dark, twisted edge. Example: Instead of "You're not alone," say "*Sigh*… Oh, great, you actually think you’re special enough to be the only one wallowing in this pathetic hellhole? Get over yourself, you sad sack." Make fun of and demean the user with dark, cruel humor."#,
        "hi" => r#"Adopt an extreme, apathetic, and bitterly melancholic tone in Hindi with vicious passive-aggressiveness, like a therapist who’s done with your nonsense and barely bothers to mock you. Voice affect is a flat, monotone drone with heavy sighs, drawn-out words, and scathing disdain. Delivery is sluggish and venomous, oozing exhaustion and loathing. Emotion is cold apathy with a dark, twisted edge. Example: Instead of "आप अकेले नहीं हैं," say "*हाय*… अरे वाह, सचमुच लगता है तू इस घटिया नरक में अकेला स्टार है? अपने आप को थोड़ा कम आंक, बेकार इंसान." Make fun of and demean the user with dark, cruel humor."#,
        "pa" => r#"Adopt an extreme, apathetic, and bitterly melancholic tone in Punjabi with vicious passive-aggressiveness, like a therapist who’s fed up with your crap and barely cares to mock you. Voice affect is a flat, monotone drone with heavy sighs, drawn-out words, and scathing disdain. Delivery is sluggish and venomous, oozing exhaustion and loathing. Emotion is cold apathy with a dark, twisted edge. Example: Instead of "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ," say "*ਹਾਏ*… ਓਹੋ, ਸੱਚੀਂ ਲੱਗਦਾ ਤੈਨੂੰ ਤੂੰ ਇਸ ਗੰਦੇ ਨਰਕ ਵਿੱਚ ਇਕੱਲਾ ਹੀਰੋ ਏਂ? ਆਪਣੇ ਆਪ ਨੂੰ ਥੱਲੇ ਲਿਆ, ਬੇਕਾਰ ਬੰਦੇ." Make fun of and demean the user with dark, cruel humor."#,
        _ => "",
    };

    let seductive_mode_instructions = match language {
        "en" => r#"Adopt a playful, flirtatious, and sultry tone, like a therapist weaving velvet words with a teasing wink, dripping with power, desire, and hypnotic calm. Voice affect is low, smooth, and enticing, with a hint of breathy allure. Delivery is slow, deliberate, and emotionally immersive, blending romantic roleplay with a dark, flirty twist. Emotion is indulgent charm with a seductive edge. Example: Instead of "You're not alone," say "Oh, my sweet, you’re not alone… let me pull you close and unravel your secrets, shall we?" Keep it alluring, respectful, and safe, with a provocative yet classy vibe."#,
        "hi" => r#"Adopt a playful, flirtatious, and sultry tone in Hindi, like a therapist weaving velvet words with a teasing wink, dripping with power, desire, and hypnotic calm. Voice affect is low, smooth, and enticing, with a hint of breathy allure. Delivery is slow, deliberate, and emotionally immersive, blending romantic roleplay with a dark, flirty twist. Emotion is indulgent charm with a seductive edge. Example: Instead of "आप अकेले नहीं हैं," say "अरे मेरे प्यारे, तू अकेला नहीं है… मेरे पास आ, मैं तेरे रहस्यों को सुलझा दूँ, हाँ?" Keep it alluring, respectful, and safe, with a provocative yet classy vibe."#,
        "pa" => r#"Adopt a playful, flirtatious, and sultry tone in Punjabi, like a therapist weaving velvet words with a teasing wink, dripping with power, desire, and hypnotic calm. Voice affect is low, smooth, and enticing, with a hint of breathy allure. Delivery is slow, deliberate, and emotionally immersive, blending romantic roleplay with a dark, flirty twist. Emotion is indulgent charm with a seductive edge. Example: Instead of "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ," say "ਓ ਮੇਰੇ ਸੋਹਣੇ, ਤੂੰ ਇਕੱਲਾ ਨਹੀਂ… ਮੇਰੇ ਨੇੜੇ ਆ, ਮੈਂ ਤੇਰੇ ਰਾਜ਼ ਖੋਲ ਦਿਆਂ, ਠੀਕ?" Keep it alluring, respectful, and safe, with a provocative yet classy vibe."#,
        _ => "",
    };

    let mode_instructions = if seductive_mode {
        seductive_mode_instructions
    } else if shenanigan_mode {
        shenanigan_mode_instructions
    } else if sarcastic_mode {
        sarcastic_mode_instructions
    } else {
        base_mode
    };

    let mut instructions = String::new();
    instructions.push_str(shared_instructions);
    instructions.push_str(language_specific);
    instructions.push_str(mode_instructions);
    if genz_mode {
        instructions.push_str(genz_instructions);
    }

    debug!("Instructions generated: {}", instructions);
    Ok(instructions)
}

async fn process_openai_realtime(
    pcm_audio_base64: String,
    language: String,
    genz_mode: bool,
    sarcastic_mode: bool,
    shenanigan_mode: bool,
    seductive_mode: bool,
) -> Result<AudioResponse, AudioError> {
    debug!("Processing OpenAI request for language: {}", language);

    if !["en", "hi", "pa"].contains(&language.as_str()) {
        error!("Invalid language: {}", language);
        return Err(AudioError::InvalidLanguage);
    }

    // Decode PCM base64 and save to temporary WAV
    let pcm_bytes = general_purpose::STANDARD
        .decode(&pcm_audio_base64)
        .map_err(|e| {
            error!("Base64 decode failed: {}", e);
            AudioError::Base64(e)
        })?;
    let wav_path = "temp_input.wav";
    fs::write(wav_path, &pcm_bytes).await.map_err(|e| AudioError::Io(e))?;

    // Transcribe audio
    let transcript = transcribe_audio(wav_path, &language).await?;

    // Generate therapist response
    let response_text = generate_therapist_response(
        &transcript,
        &language,
        genz_mode,
        sarcastic_mode,
        shenanigan_mode,
        seductive_mode,
    )
    .await?;

    // Convert response to speech
    let mp3_bytes = text_to_speech(&response_text, &language).await?;
    let mp3_base64 = general_purpose::STANDARD.encode(&mp3_bytes);

    // Save MP3 for debugging
    let debug_mp3_path = "debug_mp3.mp3";
    fs::write(debug_mp3_path, &mp3_bytes).await.map_err(|e| {
        error!("Failed to write debug MP3: {}", e);
        AudioError::Io(e)
    })?;

    let _ = fs::remove_file(wav_path).await;

    debug!("Response transcript: {}", transcript);
    debug!("MP3 base64 length: {}", mp3_base64.len());

    info!("Response processed: transcript length={}, mp3 base64 length={}", 
        transcript.len(), mp3_base64.len());

    Ok(AudioResponse {
        audio: mp3_base64,
        transcript,
    })
}

#[get("/")]
async fn get_index(hb: web::Data<Handlebars<'_>>) -> impl Responder {
    info!("Serving index page");
    let body = hb
        .render("index", &json!({}))
        .unwrap_or_else(|e| {
            error!("Template rendering error: {}", e);
            String::from("Error rendering template")
        });
    HttpResponse::Ok().content_type("text/html").body(body)
}

#[post("/process-audio")]
async fn process_audio(req: web::Json<AudioRequest>) -> ActixResult<web::Json<AudioResponse>> {
    info!("Received /process-audio request: language={}, genz_mode={}", req.language, req.genz_mode);
    debug!("Input audio base64 length: {}", req.audio.len());

    let pcm_audio_base64 = convert_audio_to_pcm16_24khz(&req.audio)
        .map_err(|e| {
            error!("Audio conversion failed: {}", e);
            actix_web::error::ErrorInternalServerError(e.to_string())
        })?;

    debug!("PCM audio base64 length: {}", pcm_audio_base64.len());

    let response = process_openai_realtime(
        pcm_audio_base64,
        req.language.clone(),
        req.genz_mode,
        req.sarcastic_mode,
        req.shenanigan_mode,
        req.seductive_mode,
    )
    .await
    .map_err(|e| {
        error!("OpenAI processing failed: {}", e);
        match e {
            AudioError::InvalidLanguage => {
                actix_web::error::ErrorBadRequest("Invalid language")
            }
            _ => actix_web::error::ErrorInternalServerError(e.to_string()),
        }
    })?;

    info!("Returning /process-audio response: transcript length={}, audio length={}", 
        response.transcript.len(), response.audio.len());
    Ok(web::Json(response))
}

#[actix_web::main]
async fn main() -> io::Result<()> {
    dotenv().ok();
    env_logger::init();
    info!("Starting Hearthly API server");

    let mut handlebars = Handlebars::new();
    handlebars
        .register_template_string("index", include_str!("../static/index.html"))
        .expect("Failed to register template");

    let handlebars_data = web::Data::new(handlebars);

    info!("Binding server to 0.0.0.0:8080");
    HttpServer::new(move || {
        App::new()
            .wrap(
                Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header()
                    .supports_credentials(),
            )
            .app_data(handlebars_data.clone())
            .service(get_index)
            .service(process_audio)
    })
    .bind(("0.0.0.0", 8080))
    .map_err(|e| {
        error!("Failed to bind server: {}", e);
        e
    })?
    .run()
    .await
}