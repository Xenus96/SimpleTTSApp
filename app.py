# -------------------------------------------------------------------------
# TTS Studio
# Copyright © 2026 CyberChief. All rights reserved.
#
# This software is licensed for Non-Commercial use only.
# See LICENSE file for details.
# -------------------------------------------------------------------------
import os
import sys

# --- [NEW] CUSTOM MODEL PATH SETUP ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the 'models' folder inside the main app folder
MODELS_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Force libraries to download models here
os.environ["TTS_HOME"] = MODELS_DIR          # For Coqui XTTS
os.environ["HF_HOME"] = MODELS_DIR           # For Transformers (MMS/Qwen)
os.environ["XDG_CACHE_HOME"] = MODELS_DIR    # Linux fallback

# CRITICAL FIX: Disable typeguard source inspection for PyInstaller
os.environ["TYPEGUARD_DISABLE"] = "true"

import types

fake_typeguard = types.ModuleType("typeguard")

def no_op_decorator(*args, **kwargs):
    def wrapper(func):
        return func
    return wrapper

fake_typeguard.typechecked = no_op_decorator
fake_typeguard.check_type = lambda *a, **k: None
fake_typeguard.TypeCheckError = Exception

sys.modules["typeguard"] = fake_typeguard

import gc
import io
import json
import re
import shutil
import threading
import time
import warnings
from threading import Thread
from tkinter import filedialog
import customtkinter as ctk
import docx
import langid
import numpy as np
import psutil
import pygame
import scipy.io.wavfile as wavfile
import torch


def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        # Running as .exe
        base_path = sys._MEIPASS
    else:
        # Running as .py
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

# --- NEW IMPORTS FOR CHATBOT ---
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- NEW IMPORTS FOR MMS ---
from transformers import VitsModel
import transformers

# 1. Suppress the specific warning about weight initialization
transformers.logging.set_verbosity_error()

# 2. (Optional) Set it back to default if you want other warnings later
transformers.logging.set_verbosity_warning()


class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        if self.widget.winfo_exists():
            self.widget.configure(state="normal")
            self.widget.insert("end", str)
            self.widget.see("end")
            self.widget.configure(state="disabled")

    def flush(self):
        pass


# --- 1. CONFIGURATION & SETUP ---
warnings.filterwarnings("ignore")
os.environ["COQUI_TOS_AGREED"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- RUNNING ON {DEVICE.upper()} ---")

USE_DEEPSPEED = False
try:
    import deepspeed

    USE_DEEPSPEED = True
    print("--- DEEPSPEED DETECTED & ENABLED ---")
except ImportError:
    print("--- DEEPSPEED NOT FOUND (Using Standard Torch) ---")

from TTS.api import TTS

# Global settings
LANGUAGES = {
    "Auto-Detect": "auto",
    "English": "en",
    "Russian": "ru",
    "German": "de",
    "Romanian": "ro",  # Now routed to MMS
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko"
}

if getattr(sys, 'frozen', False):
    # If running as EXE, use the folder where the EXE is located
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # If running as script, use the script's folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VOICES_DIR = os.path.join(BASE_DIR, "voices")
STARRED_VOICES_FILE = os.path.join(BASE_DIR, "starred_voices.json")


def smart_split(text, max_len=200):
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\n)\s'
    sentences = re.split(pattern, text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_len:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# --- POPUP WINDOW CLASS ---
class VoiceImportPopup(ctk.CTkToplevel):
    def __init__(self, parent, voices_dir, on_success_callback):
        super().__init__(parent)
        self.voices_dir = voices_dir
        self.on_success = on_success_callback

        self.title("Import Voice Reference")
        self.geometry("500x400")
        self.resizable(False, False)

        self.transient(parent)
        self.grab_set()

        title_lbl = ctk.CTkLabel(self, text="Voice Cloning Guidelines", font=("Arial", 20, "bold"))
        title_lbl.pack(pady=(20, 10))

        info_text = (
            "To get the best quality voice clone, please ensure your\n"
            "audio file meets the following criteria:\n\n"
            "1. Length: 6 to 30 seconds (ideal is ~10s).\n"
            "2. Content: Continuous speech. Avoid long silences.\n"
            "3. Quality: Clean audio with NO background music or noise.\n"
            "4. Format: .WAV (preferred), .MP3, or .FLAC.\n"
            "5. Emotion: The clone will mimic the tone of this sample.\n"
        )

        info_lbl = ctk.CTkLabel(self, text=info_text, font=("Arial", 14), justify="left", text_color="#dddddd")
        info_lbl.pack(pady=10, padx=20)

        warn_lbl = ctk.CTkLabel(self, text="⚠️ Noisy audio will result in a noisy voice clone!",
                                font=("Arial", 12, "bold"), text_color="#FF5555")
        warn_lbl.pack(pady=(0, 20))

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", pady=10, padx=20)

        self.btn_cancel = ctk.CTkButton(btn_frame, text="Cancel", fg_color="#555555", hover_color="#333333",
                                        command=self.destroy)
        self.btn_cancel.pack(side="left", expand=True, padx=5)

        self.btn_select = ctk.CTkButton(btn_frame, text="Select Audio File", fg_color="#00AA00", hover_color="#006600",
                                        command=self.browse_file)
        self.btn_select.pack(side="right", expand=True, padx=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(parent=self, filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
        if file_path:
            try:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.voices_dir, filename)
                shutil.copy(file_path, dest_path)
                self.on_success(filename)
                self.destroy()
            except Exception as e:
                print(f"Error importing voice: {e}")


class TextBlock(ctk.CTkFrame):
    def __init__(self, master, delete_command, index):
        super().__init__(master)
        self.pack(fill="x", pady=5, padx=5)
        self.index = index

        self.default_height = 250
        self.max_height = 450
        self.min_height = 250
        self.start_y = 0
        self.start_h = 0

        # Main Header
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=5, pady=2)

        self.lbl_index = ctk.CTkLabel(self.header_frame, text=f"Block {index}", font=("Arial", 12, "bold"))
        self.lbl_index.pack(side="left")

        self.btn_del = ctk.CTkButton(self.header_frame, text="X", width=30, fg_color="#FF5555", hover_color="#AA0000",
                                     command=lambda: delete_command(self))
        self.btn_del.pack(side="right")

        # Tabs
        self.tab_view = ctk.CTkTabview(self, height=self.default_height)
        self.tab_view.pack(fill="both", expand=True, padx=5, pady=0)

        self.tab_view.add("Write")
        self.tab_view.add("Listen")
        self.tab_view.set("Write")

        # --- TAB 1: WRITE ---
        self.lang_var = ctk.StringVar(value="Auto-Detect")
        self.lang_menu = ctk.CTkOptionMenu(self.tab_view.tab("Write"), variable=self.lang_var,
                                           values=list(LANGUAGES.keys()),
                                           width=120)
        self.lang_menu.pack(pady=5, anchor="w")

        # [CHANGE] Enabled Undo functionality
        self.text_area = ctk.CTkTextbox(self.tab_view.tab("Write"), height=100, undo=True)
        self.text_area.pack(fill="both", expand=True)

        # --- TAB 2: LISTEN ---
        self.listen_frame = ctk.CTkFrame(self.tab_view.tab("Listen"), fg_color="transparent")
        self.listen_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Controls Row
        self.controls_frame = ctk.CTkFrame(self.listen_frame, fg_color="transparent")
        self.controls_frame.pack(fill="x", pady=5)

        self.btn_play = ctk.CTkButton(self.controls_frame, text="▶", width=40, height=40,
                                      font=("Arial", 20),
                                      state="disabled", fg_color="gray", command=self.toggle_play)
        self.btn_play.pack(side="left", padx=(0, 10))

        # --- VOLUME SLIDER ---
        self.vol_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.vol_frame.pack(side="left", padx=10)

        ctk.CTkLabel(self.vol_frame, text="Volume", font=("Arial", 10)).pack(side="top")
        # Added command=self.update_live_volume to react to movement
        self.vol_slider = ctk.CTkSlider(self.vol_frame, from_=0, to=2.0, width=100, number_of_steps=20,
                                        command=self.update_live_volume)
        self.vol_slider.set(1.0)  # Default 100%
        self.vol_slider.pack(side="bottom")
        self.vol_slider.configure(state="disabled")  # Initially disabled

        self.btn_save_block = ctk.CTkButton(self.controls_frame, text="Save This", width=80,
                                            state="disabled", command=self.save_block_audio)
        self.btn_save_block.pack(side="right", padx=5)


        # Waveform & Seek
        self.waveform_canvas = ctk.CTkCanvas(self.listen_frame, height=50, bg="#2b2b2b", highlightthickness=0)
        self.waveform_canvas.pack(fill="both", expand=True, pady=(5, 0))

        self.waveform_canvas.bind("<Configure>", self.on_canvas_resize)

        self.time_frame = ctk.CTkFrame(self.listen_frame, fg_color="transparent")
        self.time_frame.pack(fill="x", pady=0)
        self.lbl_curr_time = ctk.CTkLabel(self.time_frame, text="00:00", font=("Arial", 10), text_color="gray")
        self.lbl_curr_time.pack(side="left")
        self.lbl_total_time = ctk.CTkLabel(self.time_frame, text="00:00", font=("Arial", 10), text_color="gray")
        self.lbl_total_time.pack(side="right")

        self.seek_slider = ctk.CTkSlider(self.listen_frame, from_=0, to=100, command=self.on_seek_drag,
                                         state="disabled")
        self.seek_slider.set(0)
        self.seek_slider.pack(fill="x", pady=0)

        # --- RESIZE GRIP ---
        self.grip = ctk.CTkFrame(self, height=8, cursor="sb_v_double_arrow", fg_color="#333333")
        self.grip.pack(fill="x", side="bottom")
        self.grip.bind("<Button-1>", self.start_resize)
        self.grip.bind("<B1-Motion>", self.do_resize)

        # Audio Logic Data
        self.audio_data = None
        self.is_playing = False
        self.audio_duration = 0.0
        self.playback_start_time = 0.0
        self.dragging = False

    def update_live_volume(self, value):
        """Changes volume dynamically during playback."""
        # Since we are applying volume to the array, we can't change amplification
        # live without reloading the array (which causes stutter).
        # Ideally, for live sliding, we just use mixer volume up to 1.0.
        # Users will need to restart playback to hear >1.0 volume changes effectively
        # or you keep this for <1.0 attenuation.
        if self.is_playing:
            pygame.mixer.music.set_volume(min(value, 1.0))

    def on_canvas_resize(self, event):
        if self.audio_data is not None:
            self.draw_waveform(self.audio_data)

    def start_resize(self, event):
        self.start_y = event.y_root
        self.start_h = self.tab_view.winfo_height()

    def do_resize(self, event):
        delta = event.y_root - self.start_y
        new_h = self.start_h + delta
        if new_h < self.min_height: new_h = self.min_height
        if new_h > self.max_height: new_h = self.max_height
        self.tab_view.configure(height=new_h)

    def get_data(self):
        content = self.text_area.get("1.0", "end").strip()
        return {
            "text": content,
            "lang": LANGUAGES[self.lang_var.get()]
        }

    def set_audio(self, audio_data):
        self.audio_data = audio_data
        self.prepare_audio_processing()

        self.btn_play.configure(state="normal", fg_color="#1f6aa5", text="▶")
        self.btn_save_block.configure(state="normal")
        self.draw_waveform(audio_data)

        self.seek_slider.configure(state="normal")
        self.seek_slider.set(0)

        # Enable Volume Slider
        self.vol_slider.configure(state="normal")

        self.lbl_curr_time.configure(text="00:00")

    def prepare_audio_processing(self):
        if self.audio_data is None: return
        self.audio_duration = len(self.audio_data) / 24000.0
        m, s = divmod(int(self.audio_duration), 60)
        self.lbl_total_time.configure(text=f"{m:02d}:{s:02d}")

    def draw_waveform(self, audio_data):
        self.waveform_canvas.delete("all")
        w = self.waveform_canvas.winfo_width()
        h = self.waveform_canvas.winfo_height()

        if w < 10: w = 400
        if h < 10: h = 50

        step = max(1, len(audio_data) // w)
        amplitudes = np.abs(audio_data[::step])
        if np.max(amplitudes) > 0:
            amplitudes = amplitudes / np.max(amplitudes)

        center_y = h / 2
        scale = (h / 2) * 0.9

        for x, amp in enumerate(amplitudes):
            if x >= w: break
            height = amp * scale
            self.waveform_canvas.create_line(x, center_y - height, x, center_y + height, fill="#1f6aa5", width=1)

    def save_block_audio(self):
        if self.audio_data is None: return
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Audio", "*.wav")])
        if save_path:
            final = self.audio_data
            peak = np.max(np.abs(final))
            if peak > 0: final = (final / peak) * 0.9
            final_clipped = np.clip(final, -1.0, 1.0)

            wavfile.write(save_path, 24000, (final_clipped * 32767).astype(np.int16))

    def toggle_play(self):
        if self.is_playing:
            self.stop_playback()
        else:
            start_pos_percent = self.seek_slider.get()
            start_sec = (start_pos_percent / 100.0) * self.audio_duration
            self.play_segment(start_time=start_sec)

    def stop_playback(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.btn_play.configure(text="▶")

    def play_segment(self, start_time=0.0):
        if self.audio_data is not None:
            try:
                # [FIX] Apply Volume Calculation Here (Digital Gain)
                vol = self.vol_slider.get()

                # Multiply data by volume (allows >1.0 amplification)
                audio_vol = self.audio_data * vol

                # [FIX] Soft Clip to prevent harsh distortion
                audio_vol = np.clip(audio_vol, -1.0, 1.0)

                # Convert to int16
                audio_int16 = (audio_vol * 32767).astype(np.int16)

                virtual_file = io.BytesIO()
                wavfile.write(virtual_file, 24000, audio_int16)
                virtual_file.seek(0)

                pygame.mixer.music.unload()
                pygame.mixer.music.load(virtual_file)

                # Set mixer volume to 1.0 (max) because we already applied gain to the data
                pygame.mixer.music.set_volume(1.0)

                if start_time > 0:
                    start_sample = int(start_time * 24000)
                    if start_sample < len(audio_int16):
                        audio_slice = audio_int16[start_sample:]
                        virtual_file = io.BytesIO()
                        wavfile.write(virtual_file, 24000, audio_slice)
                        virtual_file.seek(0)
                        pygame.mixer.music.load(virtual_file)
                        # Set the initial volume based on the slider before playing
                        current_vol = self.vol_slider.get()

                pygame.mixer.music.play()

                self.playback_start_time = time.time() - start_time
                self.is_playing = True
                self.btn_play.configure(text="■")
                self.update_seeker()
            except Exception as e:
                print(f"Playback Error: {e}")

    def on_seek_drag(self, value):
        self.dragging = True
        current_sec = (value / 100.0) * self.audio_duration
        m, s = divmod(int(current_sec), 60)
        self.lbl_curr_time.configure(text=f"{m:02d}:{s:02d}")

        if self.is_playing:
            self.play_segment(start_time=current_sec)
        self.dragging = False

    def update_seeker(self):
        if not self.is_playing: return

        if not pygame.mixer.music.get_busy():
            self.is_playing = False
            self.btn_play.configure(text="▶")
            self.seek_slider.set(0)
            self.lbl_curr_time.configure(text="00:00")
            return

        current_time = (time.time() - self.playback_start_time)
        if current_time > self.audio_duration: current_time = self.audio_duration

        if not self.dragging:
            percent = (current_time / self.audio_duration) * 100
            self.seek_slider.set(percent)
            m, s = divmod(int(current_time), 60)
            self.lbl_curr_time.configure(text=f"{m:02d}:{s:02d}")

        self.after(100, self.update_seeker)


class ChatAssistantPanel(ctk.CTkFrame):
    def __init__(self, parent, app_ref):
        super().__init__(parent, width=320, fg_color="#2b2b2b")
        self.app = app_ref
        self.pack_propagate(False)

        # 1. Header
        self.header = ctk.CTkFrame(self, height=40, fg_color="#1a1a1a", corner_radius=0)
        self.header.pack(fill="x")

        self.lbl_title = ctk.CTkLabel(
            self.header,
            text="✨ AI Companion",
            font=("Segoe UI", 16, "bold"),
            text_color="#e0e0e0"
        )
        self.lbl_title.pack(pady=8)

        # 2. Chat History Area (Scrollable)
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # 3. Input Area
        self.input_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", height=60, corner_radius=0)
        self.input_frame.pack(fill="x", side="bottom")

        self.txt_input = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="Ask to write or translate text...",
            height=35,
            font=("Segoe UI", 13),
            border_color="#444",
            fg_color="#333"
        )
        self.txt_input.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        self.txt_input.bind("<Return>", self.send_message)

        self.btn_send = ctk.CTkButton(
            self.input_frame,
            text="➤",
            width=40,
            height=35,
            fg_color="#5B2C6F",
            hover_color="#4A235A",
            command=self.send_message
        )
        self.btn_send.pack(side="right", padx=(0, 10), pady=10)

        # Model Variables
        self.model = None
        self.tokenizer = None
        # Using Qwen2.5 3B Instruct - Significantly smarter
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.is_loading = False
        self.generating = False

        # --- INITIAL GREETING ---
        self.add_message_bubble(
            "Hello! I am your personal AI Assistant. I can read your blocks, translate them, or write new content for you. Type 'help' to see the list of my capabilities.",
            is_user=False
        )

    def add_message_bubble(self, text, is_user):
        """
        Adds a message bubble to the chat with Dynamic Height.
        Returns the text widget so we can stream into it later if needed.
        """
        row_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        row_frame.pack(fill="x", pady=5)

        if is_user:
            bubble_color = "#2B78E4"
            text_color = "white"
            side = "right"
        else:
            bubble_color = "#3E3E3E"
            text_color = "#e0e0e0"
            side = "left"

        bubble = ctk.CTkFrame(row_frame, fg_color=bubble_color, corner_radius=15)
        bubble.pack(side=side, padx=5)

        # Initialize Textbox
        msg_box = ctk.CTkTextbox(
            bubble,
            width=230,
            height=10,
            font=("Segoe UI", 13),
            text_color=text_color,
            fg_color="transparent",
            wrap="word",
            activate_scrollbars=False
        )
        msg_box.pack(padx=10, pady=5)

        msg_box.insert("1.0", text)

        # Adjust Height
        self.resize_bubble(msg_box, text)
        msg_box.configure(state="disabled")

        self.after(50, self._scroll_to_bottom)
        return msg_box

    def resize_bubble(self, msg_box, text):
        msg_box.update_idletasks()
        try:
            count_res = msg_box._textbox.count("1.0", "end", "displaylines")
            if count_res:
                num_lines = int(count_res[0])
            else:
                num_lines = 1
        except:
            num_lines = text.count('\n') + (len(text) // 30) + 1

        new_height = (num_lines * 20) + 10
        msg_box.configure(height=new_height)

    def _scroll_to_bottom(self):
        self.scroll_frame._parent_canvas.yview_moveto(1.0)

    def send_message(self, event=None):
        if self.generating: return
        user_text = self.txt_input.get().strip()
        if not user_text: return

        self.txt_input.delete(0, "end")
        self.add_message_bubble(user_text, is_user=True)

        # --- [NEW] HELP COMMAND CHECK ---
        if user_text.lower() == "help":
            help_text = (
                "🤖 **AI Assistant Commands:**\n\n"
                "1. **Generation:** 'Generate text and put it inside block 1'\n"
                "2. **Translation:** 'Translate block 2 to Spanish'\n"
                "3. **Rewriting:** 'Make block 3 more formal'\n"
                "4. **Expansion:** 'Continue the story in block 4'\n\n"
                "**Note:** I can read all your current blocks to provide context!"
            )
            self.add_message_bubble(help_text, is_user=False)
            return
        # --------------------------------

        if not self.model and not self.is_loading:
            threading.Thread(target=self.load_and_reply, args=(user_text,)).start()
        elif self.model:
            threading.Thread(target=self.generate_reply, args=(user_text,)).start()

    def load_model(self):
        if self.model: return
        self.is_loading = True

        # We can create a temporary loading bubble
        self.app.safe_update(self.add_message_bubble, "[System]: Loading Qwen2 Model... (This runs once)", False)

        try:
            torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=DEVICE
            )
        except Exception as e:
            print(f"Error loading model: {e}")
        finally:
            self.is_loading = False

    def load_and_reply(self, user_text):
        self.load_model()
        self.generate_reply(user_text)

    def generate_reply(self, user_text):
        if not self.model: return
        self.generating = True

        # [CHANGE] Gather Context from Blocks
        context_str = ""
        for i, block in enumerate(self.app.blocks):
            txt = block.text_area.get("1.0", "end").strip()
            if txt:
                context_str += f"[BLOCK {block.index} CONTENT]: {txt}\n"

        # [CHANGE] Improved Prompt
        system_prompt = (
            "You are an intelligent assistant for a TTS app. "
            "You have access to the user's text blocks:\n"
            f"{context_str}\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. If the user asks to modify, translate, or correct a block, you MUST rewrite the COMPLETE text of that block.\n"
            "2. Even if they only want to change one sentence, output the whole block text merged with the changes.\n"
            "3. Use the format: [UPDATE BLOCK X: <full new content>].\n"
            "4. If translating, keep untranslated parts in their original language if requested.\n"
            "5. Do NOT use the command format if you are just chatting."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)

        # [CHANGE] Setup Streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=1024, temperature=0.7)

        # Create the UI bubble for the assistant's reply
        bubble_widget = [None]  # Mutable container to hold the widget ref

        def create_bubble():
            bubble_widget[0] = self.add_message_bubble("", is_user=False)

        self.app.safe_update(create_bubble)

        # Start generation in a separate thread so the streamer loop doesn't block
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        accumulated_text = ""

        # [CHANGE] Streaming Loop
        for new_text in streamer:
            accumulated_text += new_text

            # Safely update UI
            def update_text(w=bubble_widget[0], t=accumulated_text):
                if w:
                    w.configure(state="normal")
                    w.delete("1.0", "end")
                    w.insert("1.0", t)
                    self.resize_bubble(w, t)
                    w.configure(state="disabled")
                    self._scroll_to_bottom()

            # Wait for bubble creation
            while bubble_widget[0] is None:
                time.sleep(0.05)

            self.app.safe_update(update_text)

        # [CHANGE] Process any block update commands found in the final text
        self.process_commands(accumulated_text)
        self.generating = False

    def process_commands(self, text):
        # Look for [UPDATE BLOCK X: content]
        # Regex explanation:
        # \[UPDATE BLOCK (\d+): matches the start tag and captures the number
        # \s* optional space
        # (.*?) captures the content non-greedily
        # \] matches the closing bracket
        pattern = r"\[UPDATE BLOCK (\d+):\s*(.*?)\]"
        matches = re.findall(pattern, text, re.DOTALL)

        for index_str, new_content in matches:
            try:
                idx = int(index_str)
                target_block = next((b for b in self.app.blocks if b.index == idx), None)
                if target_block:
                    def update_ui(b=target_block, c=new_content):
                        # This enables the "Undo" because we are just inserting text programmatically
                        b.text_area.delete("1.0", "end")
                        b.text_area.insert("end", c.strip())
                        b.tab_view.set("Write")  # Switch tab so user sees change

                    self.app.safe_update(update_ui)
            except Exception as e:
                print(f"Failed to update block: {e}")


class TTSApp(ctk.CTk):
    def toggle_terminal(self):
        if not self.terminal_visible:
            self.terminal_frame.pack(side="bottom", fill="x", pady=(10, 0))
            self.terminal_toggle_btn.configure(text="Close Terminal Logs")
            self.terminal_visible = True
        else:
            self.terminal_frame.pack_forget()
            self.terminal_toggle_btn.configure(text="Open Terminal Logs")
            self.terminal_visible = False

    def crossfade_concat(self, chunks, crossfade_len=200):
        # Simple concatenation with smoothing to prevent clicks
        if not chunks: return np.array([])
        final = chunks[0]
        for i in range(1, len(chunks)):
            nxt = chunks[i]
            # Very simple concatenation: just ensure we start at 0 if possible
            # or just smooth the join.
            # A simpler approach than full crossfade:
            # Fade out end of 'final' and fade in start of 'nxt'
            if len(final) > crossfade_len:
                final[-crossfade_len:] *= np.linspace(1, 0, crossfade_len)
            if len(nxt) > crossfade_len:
                nxt[:crossfade_len] *= np.linspace(0, 1, crossfade_len)
            final = np.concatenate((final, nxt))
        return final

    def trim_audio(self, audio, threshold=0.02, padding=0.05, sr=24000):
        try:
            if len(audio) == 0: return audio
            energy = np.abs(audio)
            mask = energy > threshold
            if not np.any(mask):
                return audio
            start_idx = np.argmax(mask)
            end_idx = len(audio) - np.argmax(mask[::-1])
            pad_samples = int(sr * padding)
            start_idx = max(0, start_idx - pad_samples)
            end_idx = min(len(audio), end_idx + pad_samples)
            return audio[start_idx:end_idx]
        except Exception as e:
            print(f"Trim Error: {e}")
            return audio

    def smooth_audio(self, audio_data, fade_len_ms=20, sr=24000):
        if len(audio_data) == 0: return audio_data
        fade_samples = int(sr * fade_len_ms / 1000)
        if len(audio_data) < 2 * fade_samples:
            fade_samples = len(audio_data) // 2

        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            audio_data[:fade_samples] *= fade_in
            audio_data[-fade_samples:] *= fade_out

        return audio_data

    def __init__(self):
        super().__init__()
        self.title("GPU Accelerated TTS Studio")
        self.geometry("1280x780")
        self.minsize(1280, 780)

        if not os.path.exists(VOICES_DIR):
            os.makedirs(VOICES_DIR)

        self.starred_voices = self.load_starred_voices()

        try:
            pygame.mixer.init(frequency=24000)
        except:
            pass

        self.tts = None
        self.mms_model = None
        self.mms_tokenizer = None

        self.blocks = []
        self.block_counter = 1
        self.stop_requested = False
        self.is_rendering = False
        self.last_rendered_audio = None
        self.import_window = None

        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(self.sidebar, text="TTS Studio (GPU)", font=("Arial", 20, "bold")).pack(pady=20)

        status_color = "#00AA00" if DEVICE == "cuda" else "#AA5500"
        ctk.CTkLabel(self.sidebar, text=f"Engine: {DEVICE.upper()}", text_color=status_color,
                     font=("Arial", 12, "bold")).pack(pady=0)

        ctk.CTkLabel(self.sidebar, text="Max VRAM Usage:").pack(pady=(20, 5))

        # [NEW] Dynamic VRAM Check & Filter
        # 1. Get Total VRAM (or System RAM if CPU)
        max_available_gb = 0
        if torch.cuda.is_available():
            try:
                # Get GPU VRAM in GB
                max_available_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except:
                max_available_gb = 4  # Fallback safety
        else:
            # Fallback to System RAM for CPU mode
            max_available_gb = psutil.virtual_memory().total / (1024 ** 3)

        # 2. Generate Options (4GB to 32GB)
        all_options = range(4, 34, 2)

        # 3. Filter: Keep only options that are actually safer than the max VRAM
        # We generally want to leave a little headroom, so strictly < max_available_gb
        valid_options = [f"{x} GB" for x in all_options if x < max_available_gb]

        # Safety: Ensure at least one option exists (e.g., for 4GB cards)
        if not valid_options:
            valid_options = [f"{int(max_available_gb)} GB"] if max_available_gb > 1 else ["4 GB"]

        # 4. Set Default Value
        # Try to set "12 GB" as default, but if hardware is lower, use the highest available.
        default_val = "12 GB"
        if 12 >= max_available_gb:
            # If 12GB is too high, pick the last (highest) valid option
            default_val = valid_options[-1]

        self.ram_var = ctk.StringVar(value=default_val)
        self.ram_menu = ctk.CTkOptionMenu(self.sidebar, variable=self.ram_var,
                                          values=valid_options)
        self.ram_menu.pack(pady=5, padx=20, fill="x")

        self.timer_label = ctk.CTkLabel(self.sidebar, text="Time: 00:00", font=("Consolas", 14))
        self.timer_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: IDLE", wraplength=200)
        self.status_label.pack(pady=10)

        ctk.CTkButton(self.sidebar, text="+ Add Block", command=self.add_block).pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text="Import Text File", command=self.import_file).pack(pady=5, padx=20, fill="x")

        ctk.CTkButton(self.sidebar, text="🤖 AI Assistant", fg_color="#5B2C6F", hover_color="#4A235A",
                      command=self.toggle_chat).pack(pady=10, padx=20, fill="x")

        # --- SPEAKER SELECTION ---
        ctk.CTkLabel(self.sidebar, text="Speaker Voice (Ref Audio):").pack(pady=(20, 5))

        self.speaker_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.speaker_frame.pack(fill="x", padx=10)

        self.speaker_combo = ctk.CTkOptionMenu(self.speaker_frame, dynamic_resizing=False,
                                               command=self.on_speaker_change)
        self.speaker_combo.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_star = ctk.CTkButton(self.speaker_frame, text="☆", width=30,
                                      fg_color="transparent", border_width=1, border_color="gray",
                                      text_color="white", hover_color="#444444",
                                      command=self.toggle_star_voice)
        self.btn_star.pack(side="left", padx=(0, 5))

        self.btn_import_voice = ctk.CTkButton(self.speaker_frame, text="Import", width=50,
                                              fg_color="#F1C40F", text_color="#000", hover_color="#D4AC0D",
                                              command=self.open_import_popup)
        self.btn_import_voice.pack(side="right")

        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray").pack(fill="x", pady=20)

        self.gen_btn = ctk.CTkButton(self.sidebar, text="RENDER FULL AUDIO", command=self.start_render, height=50,
                                     fg_color="#00AA00", hover_color="#006600")
        self.gen_btn.pack(pady=10, padx=20, fill="x")

        self.stop_btn = ctk.CTkButton(self.sidebar, text="STOP", command=self.request_stop,
                                      fg_color="#AA0000", hover_color="#770000", state="disabled")
        self.stop_btn.pack(pady=5, padx=20, fill="x")

        self.save_all_btn = ctk.CTkButton(self.sidebar, text="Save Combined Audio", command=self.save_last_audio,
                                          fg_color="#444444", state="disabled")
        self.save_all_btn.pack(pady=5, padx=20, fill="x")

        self.terminal_toggle_btn = ctk.CTkButton(self.sidebar, text="Open Terminal Logs",
                                                 fg_color="#333333", command=self.toggle_terminal)
        self.terminal_toggle_btn.pack(pady=10, padx=20, fill="x")

        # --- MAIN AREA LAYOUT ---
        self.right_main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.right_main_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.chat_panel = None

        self.center_content = ctk.CTkFrame(self.right_main_container, fg_color="transparent")
        self.center_content.pack(side="left", fill="both", expand=True)

        self.scrollable_frame = ctk.CTkScrollableFrame(self.center_content, label_text="Storyboard")
        self.scrollable_frame.pack(side="top", fill="both", expand=True)

        self.terminal_visible = False
        self.terminal_frame = ctk.CTkFrame(self.center_content, fg_color="#1a1a1a", height=200)

        self.terminal_box = ctk.CTkTextbox(self.terminal_frame, font=("Consolas", 11), text_color="#FFFFFF")
        self.terminal_box.pack(fill="both", expand=True, padx=5, pady=5)
        self.terminal_box.configure(state="disabled")

        sys.stdout = TextRedirector(self.terminal_box)
        sys.stderr = TextRedirector(self.terminal_box)

        self.add_block()

        # --- NON-COMMERCIAL NOTICE & COPYRIGHT CONTAINER ---
        notice_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        notice_frame.pack(side="bottom", fill="x", pady=(10, 15))

        # 1. Non-Commercial Warning (Orange)
        ctk.CTkLabel(
            notice_frame,
            text="⚠️ NON-COMMERCIAL USE ONLY ⚠️",
            font=("Arial", 11, "bold"),
            text_color="#F39C12"
        ).pack(side="top", pady=0)

        # Line 2: Combined Credits (The "Legal Shield")
        # Linking your name and the licenses in one tight line
        ctk.CTkLabel(
            notice_frame,
            text="Copyright © 2026 CyberChief\n"
                 "Powered by Coqui, Meta NC & Qwen Research",
            font=("Arial", 9),
            text_color="gray"
        ).pack(side="top", pady=(2, 0))

        self.refresh_voice_list()

    # --- LOGIC ---
    def toggle_chat(self):
        if self.chat_panel is None:
            self.chat_panel = ChatAssistantPanel(self.right_main_container, self)
            self.chat_panel.pack(side="right", fill="y", padx=(10, 0))
        else:
            if self.chat_panel.winfo_ismapped():
                self.chat_panel.pack_forget()
            else:
                self.chat_panel.pack(side="right", fill="y", padx=(10, 0))

    def load_starred_voices(self):
        if os.path.exists(STARRED_VOICES_FILE):
            try:
                with open(STARRED_VOICES_FILE, "r") as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_starred_voices(self):
        with open(STARRED_VOICES_FILE, "w") as f:
            json.dump(self.starred_voices, f)

    def on_speaker_change(self, choice):
        self.update_star_icon()

    def update_star_icon(self):
        current = self.speaker_combo.get()
        if current in self.starred_voices:
            self.btn_star.configure(text="★", text_color="#FFD700", fg_color="#333333")
        else:
            self.btn_star.configure(text="☆", text_color="white", fg_color="transparent")

    def toggle_star_voice(self):
        current = self.speaker_combo.get()
        if current == "No Voices Found": return

        if current in self.starred_voices:
            self.starred_voices.remove(current)
        else:
            self.starred_voices.append(current)

        self.save_starred_voices()
        self.refresh_voice_list(keep_selection=current)
        self.update_star_icon()

    def refresh_voice_list(self, keep_selection=None):
        allowed_exts = {".wav", ".mp3", ".flac"}
        voice_files = []

        if os.path.exists(VOICES_DIR):
            for f in os.listdir(VOICES_DIR):
                if os.path.splitext(f)[1].lower() in allowed_exts:
                    voice_files.append(f)

        if not voice_files:
            self.speaker_combo.configure(values=["No Voices Found"])
            self.speaker_combo.set("No Voices Found")
            self.speaker_combo.configure(state="disabled")
            return

        display_names = sorted([os.path.splitext(f)[0] for f in voice_files])

        starred = [n for n in display_names if n in self.starred_voices]
        others = [n for n in display_names if n not in self.starred_voices]

        starred.sort()
        others.sort()

        final_list = starred + others

        self.speaker_combo.configure(values=final_list)
        self.speaker_combo.configure(state="normal")

        if keep_selection and keep_selection in final_list:
            self.speaker_combo.set(keep_selection)
        elif final_list:
            self.speaker_combo.set(final_list[0])

        self.update_star_icon()

    def open_import_popup(self):
        if self.import_window is None or not self.import_window.winfo_exists():
            def on_import_success(filename):
                print(f"Imported voice: {filename}")
                name_no_ext = os.path.splitext(filename)[0]
                self.refresh_voice_list(keep_selection=name_no_ext)

            self.import_window = VoiceImportPopup(self, VOICES_DIR, on_import_success)
        else:
            self.import_window.focus()

    def safe_update(self, func, *args, **kwargs):
        self.after(0, lambda: func(*args, **kwargs))

    def check_ram_usage(self):
        # [FIX] GPU VRAM Check
        if torch.cuda.is_available():
            # Get reserved memory (memory allocated by caching allocator)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory

            # If we are using more than 85% of GPU memory, clear cache
            if (reserved / total) > 0.85:
                torch.cuda.empty_cache()
                gc.collect()

        # Keep existing CPU RAM check
        limit_gb = int(self.ram_var.get().split()[0])
        process = psutil.Process(os.getpid())
        current_ram_gb = process.memory_info().rss / (1024 ** 3)
        if current_ram_gb >= limit_gb:
            gc.collect()
            torch.cuda.empty_cache()
            return False
        return True

    def load_engines(self):
        if self.tts is None:
            self.safe_update(self.status_label.configure,
                             text="Checking/Downloading XTTS Model... (This may take time)")

            # [CHANGE] Use model_name to trigger auto-download
            # This downloads to C:\Users\User\AppData\Local\tts\ or ~/.local/share/tts
            self.tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False
            ).to(DEVICE)
            print(f"XTTS Model Loaded on {DEVICE}")

        if self.mms_model is None:
            self.safe_update(self.status_label.configure, text="Loading MMS (Romanian)...")
            # [KEEP] This is already correct for auto-downloading!
            self.mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ron")
            self.mms_model = VitsModel.from_pretrained("facebook/mms-tts-ron").to(DEVICE)
            print(f"MMS Model Loaded on {DEVICE}")

    def add_block(self):
        block = TextBlock(self.scrollable_frame, self.remove_block, self.block_counter)
        self.blocks.append(block)
        self.block_counter += 1

    def remove_block(self, block):
        block.destroy()
        self.blocks.remove(block)

    def import_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text/Word", "*.txt *.docx")])
        if not filepath: return
        text = ""
        if filepath.endswith(".docx"):
            doc = docx.Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

        if self.blocks:
            self.blocks[-1].text_area.insert("end", text)

    def request_stop(self):
        self.stop_requested = True
        self.status_label.configure(text="Stopping...")

    def start_render(self):
        self.is_rendering = True
        self.gen_btn.configure(state="disabled", text="Rendering...")
        self.stop_btn.configure(state="normal")
        self.timer_label.configure(text="Time: 00:00")
        self.start_time = time.time()

        threading.Thread(target=self.timer_logic, daemon=True).start()
        threading.Thread(target=self.render_logic, daemon=True).start()

    def timer_logic(self):
        while self.is_rendering:
            elapsed = int(time.time() - self.start_time)
            m, s = divmod(elapsed, 60)
            self.safe_update(self.timer_label.configure, text=f"Time: {m:02d}:{s:02d}")
            time.sleep(1)

    def render_logic(self):
        self.stop_requested = False
        try:
            self.load_engines()

            selected_name = self.speaker_combo.get()
            speaker_wav_path = None
            if selected_name != "No Voices Found":
                for f in os.listdir(VOICES_DIR):
                    if os.path.splitext(f)[0] == selected_name:
                        speaker_wav_path = os.path.join(VOICES_DIR, f)
                        break

            xtts_lang_map = {"zh": "zh-cn", "zh-cn": "zh-cn"}

            full_audio_pieces = []

            for i, block in enumerate(self.blocks):
                if self.stop_requested: break

                data = block.get_data()
                raw_text = data['text']
                block_target_lang = data['lang']

                if not raw_text.strip(): continue

                # [FIX] Check for Romanian Language to use MMS
                if block_target_lang == "ro":
                    try:
                        # MMS requires its own logic
                        inputs = self.mms_tokenizer(raw_text, return_tensors="pt").to(DEVICE)
                        with torch.no_grad():
                            output = self.mms_model(**inputs).waveform

                        # Move to CPU and numpy, remove batch dim
                        wav = output.cpu().numpy().squeeze()

                        # MMS output might be raw; normalize it to match XTTS volume levels
                        max_val = np.max(np.abs(wav))
                        if max_val > 0:
                            wav = wav / max_val * 0.9

                        block_audio_chunks.append(wav)
                    except Exception as e:
                        print(f"MMS Error: {e}")
                        # Fallback to XTTS if MMS fails
                        block_target_lang = "en"

                self.safe_update(self.status_label.configure, text=f"Processing Block {i + 1}...")

                block_audio_chunks = []

                # --- NEW RENDERING STRATEGY ---
                if block_target_lang != "ro":
                    # CASE A: Manual Language Selection -> Use Chunking (Reduces hallucinations)
                    if block_target_lang != "auto":
                        # "smart_split" keeps chunks large (up to 250 chars) so model has context
                        # This prevents it from hallucinating on short numbers/words
                        chunks = smart_split(raw_text, max_len=250)

                        for idx, chunk in enumerate(chunks):
                            if self.stop_requested: break
                            self.check_ram_usage()

                            # Render chunk
                            wav = self.tts.tts(text=chunk,
                                               speaker_wav=speaker_wav_path,
                                               language=block_target_lang)
                            wav = np.array(wav, dtype=np.float32)

                            block_audio_chunks.append(wav)
                            # Short silence between chunks
                            block_audio_chunks.append(np.zeros(int(24000 * 0.1), dtype=np.float32))

                    # CASE B: Auto-Detect -> Split by Sentence (Required for lang detection)
                    else:
                        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
                        sentences = re.split(pattern, raw_text)
                        sentences = [s.strip() for s in sentences if s.strip()]

                        for idx, sent in enumerate(sentences):
                            if self.stop_requested: break

                            # Detect Language
                            detected, _ = langid.classify(sent)
                            detected = xtts_lang_map.get(detected, detected)

                            # Use recognized lang or default to English
                            current_lang = detected if detected in LANGUAGES.values() else "en"

                            # Render Sentence
                            wav = self.tts.tts(text=sent,
                                               speaker_wav=speaker_wav_path,
                                               language=current_lang)
                            wav = np.array(wav, dtype=np.float32)

                            block_audio_chunks.append(wav)
                            block_audio_chunks.append(np.zeros(int(24000 * 0.3), dtype=np.float32))

                    # FINALIZE BLOCK AUDIO
                    if block_audio_chunks:
                        block_final = self.crossfade_concat(block_audio_chunks)
                        # Normalize to prevent clipping
                        max_amp = np.max(np.abs(block_final))
                        if max_amp > 0:
                            block_final = block_final / max_amp * 0.95

                        self.safe_update(block.set_audio, block_final)
                        full_audio_pieces.append(block_final)

            if self.stop_requested:
                self.safe_update(self.status_label.configure, text="Stopped.")
            elif full_audio_pieces:
                self.last_rendered_audio = np.concatenate(full_audio_pieces)
                self.safe_update(self.save_all_btn.configure, state="normal")
                self.safe_update(self.status_label.configure, text="Render Complete!")
            else:
                self.safe_update(self.status_label.configure, text="No text found.")

        except Exception as e:
            print(f"Render Error: {e}")
            import traceback
            traceback.print_exc()
            self.safe_update(self.status_label.configure, text="Error During Render")
        finally:
            self.is_rendering = False
            self.safe_update(self.gen_btn.configure, state="normal", text="RENDER FULL AUDIO")
            self.safe_update(self.stop_btn.configure, state="disabled")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    def save_last_audio(self):
        if self.last_rendered_audio is None: return
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Audio", "*.wav")])
        if save_path:
            wavfile.write(save_path, 24000, (self.last_rendered_audio * 32767).astype(np.int16))


if __name__ == "__main__":
    app = TTSApp()
    app.mainloop()