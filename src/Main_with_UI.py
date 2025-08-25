import pickle
import threading
from time import sleep

import PIL.Image, PIL.ImageDraw
import customtkinter as ctk
import os

from Main_without_UI import get_default_speakers, record_audio, convert_to_melspec, predict, WAVE_OUTPUT_FILENAME
from ResolumeBridge import is_resolume_reachable, activate_layer

MODELS_DIR = "saved_models"
MODEL_PATH = "saved_models/rock_metal.onnx"  # Default model path
TITLE = "BeatLight"
IMAGE_PATH = "assets/Cover_IMG.jpg"  # Set your image file path here
save_icon_path = "assets/save_24dp_E3E3E3_FILL0_wght400_GRAD0_opsz24.png"
load_icon_path = "assets/download_24dp_E3E3E3_FILL0_wght400_GRAD0_opsz24.png"
TIME_BETWEEN_MEASUREMENTS = 30  # Default time between measurements in seconds
AUDIO_THREAD = None
STOP_EVENT = threading.Event()
PATH_SETTINGS = "mappings.pkl"

# --- Mapping Dictionary ---
mappings = {}

# --- UI Functions ---
def on_model_select(choice):
    """Callback for model selection dropdown."""
    global MODEL_PATH
    MODEL_PATH = os.path.join(MODELS_DIR, choice)
    log_message(f"Selected model: {choice}")

def on_button_click():
    selected = model_var.get()
    log_message(f"Running with model: {selected}")

def log_message(message: str):
    logbox.insert("end", message + "\n")
    logbox.see("end")
    # update_status_lamp()  # Update lamp on every log

def update_status_lamp():
    if is_resolume_reachable():
        lamp.configure(fg_color="green")
        # log_message("Resolume is reachable")
    else:
        lamp.configure(fg_color="red")
        # log_message("Resolume is NOT reachable")

def toggle_button_text():
    if start_button.cget("text") == "":
        start_button.configure(text="Stop")
    elif start_button.cget("text") == "Stop":
        start_button.configure(text="Start")
        stop_counter()
    else:
        start_button.configure(text="Stop")
        start_counter()

def schedule_lamp_update(*args):
    update_status_lamp()
    app.after(10000, schedule_lamp_update, *args)

def on_closing():
    """Handle window close event."""
    print("Closing application...")
    STOP_EVENT.set()
    sleep(0.3) # Give the thread time to stop
    app.destroy()

def set_resolume_visual(lable):
    """Set the visual in Resolume based on the predicted label."""
    if len(mappings) == 0:
        log_message("No mappings defined. Please add mappings first.")
        return
    for layer, column in mappings.get(lable, []):
        try:
            activate_layer(layer, column)
            log_message(f"Activated Layer {layer}, Column {column} for genre '{lable}'")
        except Exception as e:
            log_message(f"Error activating layer {layer}, column {column}: {e}")

def audio_measurement_thread():
    """Thread function to handle audio measurement and genre prediction."""
    while not STOP_EVENT.is_set():
        try:
            log_message("Searching for default speakers...")
            default_speakers = get_default_speakers()
            log_message(f"Default speakers found: {default_speakers['index']}){default_speakers['name']}")
        except Exception as e:
            log_message(f"Error finding default speakers: {e}")
            return

        try:
            log_message("Recording audio...")
            record_audio(default_speakers)
        except Exception as e:
            log_message(f"Error recording audio: {e}")
            return

        try:
            log_message("Converting audio to Mel-spectrogram...")
            mel_spec = convert_to_melspec(WAVE_OUTPUT_FILENAME)
        except Exception as e:
            log_message(f"Error converting to Mel-spectrogram: {e}")
            return

        try:
            log_message("Predicting genre...")
            lable = predict(mel_spec, MODEL_PATH)
            log_message(f"Predicted genre: {lable}")
            set_resolume_visual(lable)
        except Exception as e:
            log_message(f"Error predicting genre: {e}")
            return

        log_message(f"Waiting {TIME_BETWEEN_MEASUREMENTS} for next measurement...")
        sleep(TIME_BETWEEN_MEASUREMENTS)

def start_counter():
    global AUDIO_THREAD
    if AUDIO_THREAD is None or not AUDIO_THREAD.is_alive():
        STOP_EVENT.clear()
        AUDIO_THREAD = threading.Thread(target=audio_measurement_thread)
        AUDIO_THREAD.start()

def stop_counter():
    STOP_EVENT.set()
    log_message("measurement stopped.")

def set_time_between_measurements():
    global TIME_BETWEEN_MEASUREMENTS
    try:
        if time_entry.get() == "":
            return # Do not change if entry is empty
        TIME_BETWEEN_MEASUREMENTS = int(time_entry.get())
        log_message(f"Time between measurements set to {TIME_BETWEEN_MEASUREMENTS}")
    except ValueError:
        log_message("Please enter a valid number for time.")

# --- Mapping Functions ---
def add_mapping():
    """Add or update mapping for genre → (layer, column)."""
    genre = genre_var.get()
    try:
        layer = int(layer_entry.get())
        if layer < 1 or layer > 3:
            log_message("Layer must be between 1 and 3")
            return
        column = int(column_entry.get())
        if column < 1 or column > 7:
            log_message("Column must be between 1 and 7")
            return
    except ValueError:
        log_message("Layer/Column must be numbers")
        return

    if genre not in mappings:
        mappings[genre] = []
    if len(mappings[genre]) >= 3:
        log_message(f"Max 3 mappings for {genre} reached. Clear mappings first.")
        return
    else:
        mappings[genre].append((layer, column))
        log_message(f"Mapping added: {genre} -> Layer {layer}, Column {column}")
    print(mappings)

def clear_mappings():
    genre = genre_var.get()
    mappings[genre] = []
    log_message(f"Mappings for {genre} cleared.")

# --- Icon Buttons (next to lamp) ---
def save_action():
    try:
        with open("../mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)
        log_message("Mappings saved to mappings.pkl.")
    except Exception as e:
        log_message(f"Error saving mappings: {e}")

def load_action():
    global mappings
    try:
        with open(PATH_SETTINGS, "rb") as f:
            mappings = pickle.load(f)
        log_message("Mappings loaded from mappings.pkl.")
        log_message(f"Loaded mappings: {mappings}")
    except Exception as e:
        log_message(f"Error loading mappings: {e}")

# --- Main Window ---
app = ctk.CTk()
app.title(TITLE)
app.geometry("500x500")
app.resizable(False, False)
app.protocol("WM_DELETE_WINDOW", on_closing)

# --- Layout mit grid ---
app.grid_rowconfigure(1, weight=1)   # middle frame grows
app.grid_rowconfigure(2, weight=0)   # logbox fixed bottom
app.grid_columnconfigure(0, weight=1)

# --- Top bar ---
top_frame = ctk.CTkFrame(app)
top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

label = ctk.CTkLabel(top_frame, text=TITLE, font=("Arial", 18))
label.pack(side="left", padx=10, pady=5)

lamp = ctk.CTkLabel(top_frame, text="", width=20, height=20, fg_color="red", corner_radius=10)
lamp.pack(side="right", padx=10, pady=5)

if os.path.exists(save_icon_path):
    save_img_raw = PIL.Image.open(save_icon_path).resize((24, 24))
    save_img = ctk.CTkImage(light_image=save_img_raw, size=(24, 24))
else:
    save_img = None

if os.path.exists(load_icon_path):
    load_img_raw = PIL.Image.open(load_icon_path).resize((24, 24))
    load_img = ctk.CTkImage(light_image=load_img_raw, size=(24, 24))
else:
    load_img = None

save_btn = ctk.CTkButton(top_frame, image=save_img, text="", width=30, height=30, command=save_action, fg_color="transparent", hover=True, hover_color="#474747")
save_btn.pack(side="right", padx=2, pady=5)

load_btn = ctk.CTkButton(top_frame, image=load_img, text="", width=30, height=30, command=load_action, fg_color="transparent", hover=True, hover_color="#474747")
load_btn.pack(side="right", padx=2, pady=5)

# --- Middle controls ---
middle_frame = ctk.CTkFrame(app)
middle_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

# 3 Spalten
middle_frame.grid_columnconfigure(0, weight=1)  # Genre Controls
middle_frame.grid_columnconfigure(1, weight=1)  # Model & Button
middle_frame.grid_columnconfigure(2, weight=1)  # Bild

# --- Genre Mapping Controls (linke Spalte) ---
genre_var = ctk.StringVar(value="Rock'n'Roll")
genre_dropdown = ctk.CTkOptionMenu(middle_frame, variable=genre_var,
                                   values=["Rock'n'Roll", "HardRock", "HeavyMetal",
                                           "TrashMetal", "MetalcoreMelodic", "DeathMetal", "Grindcore"])
genre_dropdown.grid(row=1, column=0, padx=10, pady=5, sticky="n")

column_entry = ctk.CTkEntry(middle_frame, placeholder_text="Column ID")
column_entry.grid(row=2, column=0, padx=10, pady=5, sticky="n")
column_entry.bind("<Return>", lambda event: add_mapping())

layer_entry = ctk.CTkEntry(middle_frame, placeholder_text="Layer ID")
layer_entry.grid(row=3, column=0, padx=10, pady=5, sticky="n")
layer_entry.bind("<Return>", lambda event: add_mapping())

add_button = ctk.CTkButton(middle_frame, text="Add Mapping", command=add_mapping)
add_button.grid(row=4, column=0, padx=10, pady=5, sticky="n")

clear_button = ctk.CTkButton(middle_frame, text="Clear Mappings", command=clear_mappings)
clear_button.grid(row=5, column=0, padx=10, pady=5, sticky="n")

time_entry = ctk.CTkEntry(middle_frame, placeholder_text="Time Between Measurements (s)")
time_entry.grid(row=6, column=0, padx=10, pady=5, sticky="n")
time_entry.bind("<Return>", lambda event: set_time_between_measurements())

# --- Model Dropdown (mittlere Spalte oben) ---
if os.path.exists(MODELS_DIR):
    models = [f for f in os.listdir(MODELS_DIR) if os.path.isfile(os.path.join(MODELS_DIR, f))]
else:
    models = []

model_var = ctk.StringVar(value=models[0] if models else "No models found")
dropdown = ctk.CTkOptionMenu(middle_frame, values=models if models else ["No models found"],
                              variable=model_var,
                              command=on_model_select)
dropdown.grid(row=0, column=0, padx=10, pady=5, sticky="n")

start_button = ctk.CTkButton(middle_frame, text="Start", command=lambda: [set_time_between_measurements(), toggle_button_text()])
start_button.grid(row=10, column=0, padx=10, pady=5, sticky="s")

# --- Image in rechte Spalte (Spalte 2) ---
if os.path.exists(IMAGE_PATH):
    img_raw = PIL.Image.open(IMAGE_PATH).convert("RGBA")
    radius = 20  # Radius der abgerundeten Ecken

    # Maske für abgerundete Ecken erstellen
    mask = PIL.Image.new("L", img_raw.size, 0)
    draw = PIL.ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), img_raw.size], radius=radius, fill=255)
    img_raw.putalpha(mask)

    img = ctk.CTkImage(light_image=img_raw, size=(300, 300))
    image_label = ctk.CTkLabel(middle_frame, image=img, text="")
    image_label.grid(row=0, column=2, rowspan=11, padx=10, pady=5, sticky="nsew")
else:
    image_label = ctk.CTkLabel(middle_frame, text="Image not found")
    image_label.grid(row=0, column=2, rowspan=11, padx=10, pady=5, sticky="nsew")

# --- Logbox (immer unten) ---
logbox = ctk.CTkTextbox(app, width=550, height=120)
logbox.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

# Initial check
schedule_lamp_update()

app.mainloop()
