import os
import sys
import warnings
import inspect
from types import ModuleType
import time
import psutil  # For RAM monitoring
import threading
import ctypes
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import docx
import numpy as np
import scipy.io.wavfile as wavfile
import pygame
import langid
import gc

# 1. SILENCE THE NOISE
warnings.filterwarnings("ignore")

# 2. SURGICAL BYPASS OF THE CRASHING INIT
import importlib.util

spec = importlib.util.find_spec('TTS')
if spec:
    m_tts = ModuleType("TTS")
    m_tts.__path__ = spec.submodule_search_locations
    m_tts.__package__ = "TTS"
    m_tts.TORCHCODEC_IMPORT_ERROR = None
    sys.modules["TTS"] = m_tts

# 3. THE ULTIMATE PATCH BLOCK
import transformers.utils.import_utils as import_utils

if not hasattr(import_utils, 'is_torch_greater_or_equal'):
    import torch
    from packaging import version


    def is_torch_greater_or_equal(target_version):
        v = torch.__version__.split('+')[0]
        return version.parse(v) >= version.parse(target_version)


    import_utils.is_torch_greater_or_equal = is_torch_greater_or_equal

if not hasattr(import_utils, 'is_torchcodec_available'):
    import_utils.is_torchcodec_available = lambda: False

try:
    import transformers.pytorch_utils as pytorch_utils
except ImportError:
    m = ModuleType("transformers.pytorch_utils")
    sys.modules["transformers.pytorch_utils"] = m
    import transformers.pytorch_utils as pytorch_utils

if not hasattr(pytorch_utils, 'isin_mps_friendly'):
    import torch

    pytorch_utils.isin_mps_friendly = torch.isin

# --- THE GLOBAL TENSOR TRUTHINESS FIX ---
import torch

original_bool = torch.Tensor.__bool__


def safe_bool(self):
    try:
        if self.numel() > 1:
            return True
        return original_bool(self)
    except:
        return True


torch.Tensor.__bool__ = safe_bool
print("Global Fix: Tensor Ambiguity resolved.")
# ----------------------------------------

import functools

original_torch_load = torch.load


def absolute_forced_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)


torch.load = absolute_forced_load
import torch.serialization

torch.serialization.load = absolute_forced_load

print("System Patched: Ironclad Torch security override applied.")

import torch.jit


def pass_through(obj, *args, **kwargs):
    return obj


torch.jit.script = pass_through


# --- 1. THE "LOBOTOMY" PATCH ---
def patched_getsource(obj): return ""


def patched_getsourcelines(obj): return [""], 0


def patched_findsource(obj): return [""], 0


inspect.getsource = patched_getsource
inspect.getsourcelines = patched_getsourcelines
inspect.findsource = patched_findsource
inspect.getsourcefile = lambda x: None

try:
    import torch._sources

    torch._sources.get_source_lines_and_file = lambda obj, error_msg=None: ([""], None, 0)
    torch._sources.parse_def = lambda func: None
    torch.jit.is_tracing = lambda: False
    torch.jit.is_scripting = lambda: False
except:
    pass

if hasattr(sys, '_MEIPASS'):
    torch_lib = os.path.join(sys._MEIPASS, "torch", "lib")
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)

os.environ["TYPEGUARD_DISABLE"] = "1"
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# ---------------------------------------------------

# --- THE SLIDING WINDOW CACHE PATCH (V7) ---
try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    import torch


    def patched_attn_sliding_window(self, query, key, value, attention_mask=None, head_mask=None):
        q_len = query.size(-2)
        k_len = key.size(-2)

        if k_len != q_len and not self.is_cross_attention:
            if k_len > q_len:
                key = key[:, :, -q_len:, :]
                value = value[:, :, -q_len:, :]
            else:
                padding = (0, 0, 0, q_len - k_len)
                key = torch.nn.functional.pad(key, padding)
                value = torch.nn.functional.pad(value, padding)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        cur_q, cur_k = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.tril(torch.ones((cur_q, cur_k), dtype=torch.bool, device=query.device)).view(1, 1, cur_q,
                                                                                                         cur_k)

        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None and attention_mask.size(-1) == cur_k:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights.to(value.dtype), value), attn_weights


    GPT2Attention._attn = patched_attn_sliding_window
    print("Sliding Window Patch: Cache mismatch resolved.")
except Exception as e:
    print(f"Sliding Window Patch Failed: {e}")
# --------------------------------------------

# --- THE NUCLEAR COMPATIBILITY PATCH (V2) ---
try:
    import transformers.modeling_utils

    transformers.modeling_utils.PreTrainedModel._validate_model_class = lambda self: None


    def dummy_prepare(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}


    transformers.modeling_utils.PreTrainedModel.prepare_inputs_for_generation = dummy_prepare
    transformers.modeling_utils.PreTrainedModel.can_generate = lambda self: True
    print("Nuclear Patch V2: Validation bypassed.")
except Exception as e:
    print(f"Nuclear Patch Failed: {e}")
# ---------------------------------------

from TTS.api import TTS
import transformers
from transformers import GPT2Config

GPT2Config.is_decoder = True
transformers.logging.set_verbosity_error()

# --- CONFIGURATION ---
MODEL_REL_PATH = "models/tts_models--multilingual--multi-dataset--xtts_v2"
LANGUAGES = {
    "Auto-Detect": "auto",
    "English": "en",
    "Russian": "ru",
    "German": "de",
    "Romanian": "ro",
    "French": "fr",
    "Spanish": "es"
}

TOP_SPEAKERS = [
    "Ana Florence",
    "Viktor Menelaos",
    "Daisy Studious",
    "Damien Black",
    "Sofia Hellen",
    "Tammie Ema",
    "Walter Steven",
    "Zhenya Karley",
    "Asya Anara"
]


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class TextBlock(ctk.CTkFrame):
    def __init__(self, master, delete_command, index):
        super().__init__(master)
        self.pack(fill="x", pady=5, padx=5)

        self.ctrl_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ctrl_frame.pack(fill="x", padx=5, pady=2)

        self.lbl_index = ctk.CTkLabel(self.ctrl_frame, text=f"Block {index}", font=("Arial", 12, "bold"))
        self.lbl_index.pack(side="left")

        self.lang_var = ctk.StringVar(value="Auto-Detect")
        self.lang_menu = ctk.CTkOptionMenu(self.ctrl_frame, variable=self.lang_var, values=list(LANGUAGES.keys()),
                                           width=100)
        self.lang_menu.pack(side="left", padx=10)

        self.btn_del = ctk.CTkButton(self.ctrl_frame, text="X", width=30, fg_color="#FF5555", hover_color="#AA0000",
                                     command=lambda: delete_command(self))
        self.btn_del.pack(side="right")

        self.text_area = ctk.CTkTextbox(self, height=80)
        self.text_area.pack(fill="x", padx=5, pady=5)

    def get_data(self):
        return {
            "text": self.text_area.get("0.0", "end").strip(),
            "lang": LANGUAGES[self.lang_var.get()]
        }


class TTSApp(ctk.CTk):
    # --- LOGIC METHODS ---
    def request_stop(self):
        self.stop_requested = True
        self.status_label.configure(text="Stopping...")

    def check_ram_usage(self):
        """
        STRICT LIMIT ENFORCER
        Returns False if RAM is critically full and cannot be cleared.
        """
        limit_gb = int(self.ram_var.get().split()[0])
        process = psutil.Process(os.getpid())

        # Check current usage
        current_ram_gb = process.memory_info().rss / (1024 ** 3)

        if current_ram_gb >= limit_gb:
            print(f"STRICT LIMIT: RAM at {current_ram_gb:.2f}GB (Limit {limit_gb}GB). Purging...")
            gc.collect()  # Force Python Garbage Collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Check again after purge
            current_ram_gb = process.memory_info().rss / (1024 ** 3)
            if current_ram_gb >= limit_gb:
                self.status_label.configure(text=f"RAM Limit Reached ({current_ram_gb:.1f}GB). Waiting...")
                time.sleep(1)  # Pause to let system settle
                return False  # Still too high

        return True  # Safe to proceed

    def update_timer(self):
        if self.is_rendering:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.timer_label.configure(text=f"Time: {mins:02d}:{secs:02d}")
            self.after(1000, self.update_timer)

    def detect_language_segments(self, text, manual_lang_code="auto"):
        words = text.split()
        if not words: return []

        segments = []
        current_chunk = []
        valid_langs = list(LANGUAGES.values())

        if manual_lang_code != "auto":
            return [(text, manual_lang_code)]

        current_lang, _ = langid.classify(words[0])
        if current_lang not in valid_langs: current_lang = "en"

        for word in words:
            det_lang, _ = langid.classify(word)
            if det_lang not in valid_langs: det_lang = "en"

            if det_lang != current_lang:
                segments.append((" ".join(current_chunk), current_lang))
                current_chunk = [word]
                current_lang = det_lang
            else:
                current_chunk.append(word)

        if current_chunk:
            segments.append((" ".join(current_chunk), current_lang))
        return segments

    def save_last_audio(self):
        if self.last_rendered_audio is None:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Audio", "*.wav")])
        if save_path:
            try:
                wavfile.write(save_path, 24000, (self.last_rendered_audio * 32767).astype(np.int16))
                self.status_label.configure(text="Saved Again!")
            except Exception as e:
                self.status_label.configure(text="Save Error!")
                print(e)

    def toggle_controls(self, is_rendering):
        """Disables/Enables UI elements based on state"""
        state = "disabled" if is_rendering else "normal"
        stop_state = "normal" if is_rendering else "disabled"

        self.gen_btn.configure(state=state)
        self.ram_menu.configure(state=state)
        self.speaker_menu.configure(state=state)
        self.stop_btn.configure(state=stop_state)

    def __init__(self):
        super().__init__()
        self.title("Multilingual TTS Studio")
        self.geometry("1280x720")

        self.tts = None
        self.blocks = []
        self.block_counter = 1
        self.stop_requested = False
        self.is_rendering = False
        self.last_rendered_audio = None
        self.loaded_speakers = ["Ana Florence"]  # Cache speakers so we don't lose them on unload

        # --- LEFT PANEL (Controls) ---
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(self.sidebar, text="TTS Studio", font=("Arial", 20, "bold")).pack(pady=20)

        # RAM LIMITER
        ctk.CTkLabel(self.sidebar, text="Max RAM Usage:").pack(pady=(10, 5))
        ram_options = [f"{i} GB" for i in range(2, 22, 2)]
        self.ram_var = ctk.StringVar(value="16 GB")
        self.ram_menu = ctk.CTkOptionMenu(self.sidebar, variable=self.ram_var, values=ram_options)
        self.ram_menu.pack(pady=5, padx=20, fill="x")

        # TIMER LABEL
        self.timer_label = ctk.CTkLabel(self.sidebar, text="Time: 00:00", font=("Consolas", 14))
        self.timer_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: IDLE (RAM Cleared)")
        self.status_label.pack(pady=10)

        ctk.CTkButton(self.sidebar, text="Add Text Block", command=self.add_block).pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text="Import Text File", command=self.import_file).pack(pady=10, padx=20, fill="x")

        self.speaker_var = ctk.StringVar(value="Ana Florence")
        ctk.CTkLabel(self.sidebar, text="Speaker Voice:").pack(pady=(20, 5))
        self.speaker_menu = ctk.CTkOptionMenu(self.sidebar, variable=self.speaker_var,
                                              values=self.loaded_speakers)
        self.speaker_menu.pack(pady=5, padx=20, fill="x")

        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray").pack(fill="x", pady=20)

        self.gen_btn = ctk.CTkButton(self.sidebar, text="RENDER AUDIO", command=self.start_render, height=50,
                                     fg_color="#00AA00", hover_color="#006600")
        self.gen_btn.pack(pady=10, padx=20, fill="x")

        self.stop_btn = ctk.CTkButton(self.sidebar, text="STOP RENDER",
                                      command=self.request_stop,
                                      fg_color="#AA0000", hover_color="#770000", state="disabled")
        self.stop_btn.pack(pady=5, padx=20, fill="x")

        self.save_again_btn = ctk.CTkButton(self.sidebar, text="Save Last Audio",
                                            command=self.save_last_audio,
                                            fg_color="#444444", state="disabled")
        self.save_again_btn.pack(pady=10, padx=20, fill="x")

        # --- RIGHT PANEL (Blocks) ---
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="Story Board")
        self.scrollable_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.add_block()
        # Initial load to populate speakers, then unload to save RAM
        threading.Thread(target=self.initial_boot_sequence, daemon=True).start()

    def initial_boot_sequence(self):
        """Loads engine once to get speakers, then unloads it"""
        self.load_engine()
        # Immediately free RAM after getting speakers
        self.unload_engine()

    def load_engine(self):
        """Loads the TTS model into memory"""
        if self.tts is not None: return  # Already loaded

        try:
            model_full_path = get_resource_path(MODEL_REL_PATH)
            config_path = os.path.join(model_full_path, "config.json")
            self.status_label.configure(text="Loading Model...")

            if os.path.exists(model_full_path):
                self.tts = TTS(model_path=model_full_path, config_path=config_path).to("cpu")
            else:
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

            if hasattr(self.tts, 'model') and hasattr(self.tts.model, 'gpt'):
                self.tts.model.gpt.generate = self.tts.model.gpt.transformer.generate
                self.tts.model.gpt.can_generate = lambda: True
                self.tts.model.gpt.prepare_inputs_for_generation = self.tts.model.gpt.transformer.prepare_inputs_for_generation
                self.tts.model.config.use_cache = True

            # Update speaker cache
            available_speakers = self.tts.speakers if self.tts.speakers else ["Ana Florence"]
            final_speakers = [s for s in TOP_SPEAKERS if s in available_speakers]
            if not final_speakers: final_speakers = available_speakers[:10]

            self.loaded_speakers = final_speakers
            self.speaker_menu.configure(values=self.loaded_speakers)

            if self.speaker_var.get() not in self.loaded_speakers:
                self.speaker_var.set(self.loaded_speakers[0])

            self.status_label.configure(text="Ready")
        except Exception as e:
            self.status_label.configure(text="Engine Error!")
            print(f"Engine Load Error: {e}")

    def unload_engine(self):
        """Destroys the TTS object to free RAM"""
        if self.tts:
            del self.tts
            self.tts = None
            gc.collect()
            self.status_label.configure(text="IDLE (RAM Cleared)")
            print("Engine Unloaded. RAM freed.")

    def add_block(self):
        block = TextBlock(self.scrollable_frame, self.remove_block, self.block_counter)
        self.blocks.append(block)
        self.block_counter += 1

    def remove_block(self, block_obj):
        block_obj.destroy()
        self.blocks.remove(block_obj)

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
        else:
            self.add_block()
            self.blocks[-1].text_area.insert("0.0", text)

    def start_render(self):
        self.gen_btn.configure(text="Rendering...")
        self.timer_label.configure(text="Time: 00:00")
        self.save_again_btn.configure(state="disabled")

        # 1. LOCK UI
        self.is_rendering = True
        self.toggle_controls(True)

        # 2. START TIMER
        self.start_time = time.time()
        self.update_timer()

        threading.Thread(target=self.render_logic, daemon=True).start()

    def render_logic(self):
        self.stop_requested = False

        try:
            # 1. DYNAMIC LOAD
            # We load the engine ONLY when rendering starts
            self.load_engine()

            audio_segments = []
            speaker = self.speaker_var.get()

            for i, block in enumerate(self.blocks):
                if self.stop_requested: break

                # --- STRICT RAM CHECK ---
                # This loop prevents the next chunk from starting if RAM is full
                while not self.check_ram_usage():
                    if self.stop_requested: break
                    print("Waiting for RAM to drop...")
                    time.sleep(2)

                data = block.get_data()
                full_text = data['text']
                block_lang = data['lang']

                if not full_text: continue

                self.status_label.configure(text=f"Analyzing Block {i + 1}...")
                segments = self.detect_language_segments(full_text, block_lang)

                for text_chunk, lang_chunk in segments:
                    if self.stop_requested: break

                    self.status_label.configure(text=f"Rendering ({lang_chunk})...")

                    # Strict check before heavy compute
                    self.check_ram_usage()

                    wav_data = np.array(self.tts.tts(text=" " + text_chunk, speaker=speaker, language=lang_chunk),
                                        dtype=np.float32)
                    audio_segments.append(wav_data)

                    # Clear intermediate tensors
                    gc.collect()

            if self.stop_requested:
                self.status_label.configure(text="Stopped by user")
                return

            if not audio_segments:
                self.status_label.configure(text="No text!")
                return

            self.status_label.configure(text="Stitching...")

            sample_rate = 24000
            overlap_samples = int(sample_rate * 0.08)
            final_wav = np.trim_zeros(audio_segments[0])

            for i in range(1, len(audio_segments)):
                next_seg = np.trim_zeros(audio_segments[i])
                if len(final_wav) < overlap_samples or len(next_seg) < overlap_samples:
                    final_wav = np.concatenate([final_wav, next_seg])
                    continue
                fade_out = np.linspace(1.0, 0.0, overlap_samples)
                fade_in = np.linspace(0.0, 1.0, overlap_samples)
                overlap_zone = (final_wav[-overlap_samples:] * fade_out) + (next_seg[:overlap_samples] * fade_in)
                final_wav = np.concatenate([final_wav[:-overlap_samples], overlap_zone, next_seg[overlap_samples:]])

            self.last_rendered_audio = final_wav
            self.save_again_btn.configure(state="normal")

            # Save Logic
            save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Audio", "*.wav")])
            if save_path:
                wavfile.write(save_path, 24000, (final_wav * 32767).astype(np.int16))
                self.status_label.configure(text="Done!")
                try:
                    pygame.mixer.init()
                    pygame.mixer.music.load(save_path)
                    pygame.mixer.music.play()
                except:
                    pass
            else:
                self.status_label.configure(text="Generated (Unsaved)")

        except Exception as e:
            self.status_label.configure(text="Error!")
            print(f"Render Error: {e}")
        finally:
            self.is_rendering = False
            self.gen_btn.configure(text="RENDER AUDIO")

            # 2. UNLOCK UI
            self.toggle_controls(False)

            # 3. DYNAMIC UNLOAD
            # Crucial: This frees the 4GB+ model from RAM immediately
            self.unload_engine()
            gc.collect()


if __name__ == "__main__":
    app = TTSApp()
    app.mainloop()