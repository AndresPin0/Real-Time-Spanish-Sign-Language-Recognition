import os, json, time
from typing import List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- TensorFlow/Keras ---
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.applications import resnet, mobilenet_v3


def detect_architecture(model) -> str:
    """Devuelve 'resnet', 'mobilenet_v3' o 'unknown' según nombres de capas/modelo."""
    tokens = []
    try:
        tokens.append((model.name or "").lower())
        tokens += [l.name.lower() for l in model.layers]
    except Exception:
        pass
    text = " ".join(tokens)
    if "resnet" in text:
        return "resnet"
    if "mobilenet" in text and "v3" in text:
        return "mobilenet_v3"
    return "unknown"

def preprocess_for(mode: str, arr: np.ndarray) -> np.ndarray:
    """
    mode ∈ {'resnet','mobilenet_v3','generic'}.
    arr: float32 RGB [0..255] sin escalar.
    """
    if mode == "resnet":
        return resnet.preprocess_input(arr.copy())       # maneja RGB→BGR y mean-sub
    if mode == "mobilenet_v3":
        return mobilenet_v3.preprocess_input(arr.copy()) # escala a [-1,1]
    # genérico: [0,1]
    return (arr.astype(np.float32) / 255.0)

class RealtimeSignApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Predicción en tiempo real – Modelo .h5 (ResNet/MobileNetV3)")
        self.root.geometry("1024x740")

        # Estado
        self.model = None
        self.model_path: Optional[str] = None
        self.labels: Optional[List[str]] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame = None
        self.input_shape: Optional[Tuple[int, int, int]] = None  # (H,W,C)
        self.camera_index = tk.IntVar(value=0)
        self.prob_threshold = tk.DoubleVar(value=0.0)
        self.fps_value = tk.StringVar(value="-")
        self.pred_text = tk.StringVar(value="Cargue un modelo .h5 para empezar")
        self.arch_detected = tk.StringVar(value="unknown")
        self.preproc_choice = tk.StringVar(value="Auto")  # Auto/ResNet/MobileNetV3/Genérico

        self._build_ui()

    # ---------------------- UI ----------------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Cargar modelo (.h5)", command=self.load_model_h5).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Cargar labels (.txt/.json)", command=self.load_labels).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Cargar imagen", command=self.load_image_and_predict).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Cámara:").pack(side=tk.LEFT, padx=(12, 2))
        cam_spin = ttk.Spinbox(top, from_=0, to=10, width=5, textvariable=self.camera_index)
        cam_spin.pack(side=tk.LEFT)

        ttk.Label(top, text="Umbral prob.:").pack(side=tk.LEFT, padx=(12, 2))
        thr = ttk.Scale(top, from_=0.0, to=0.9, orient=tk.HORIZONTAL, variable=self.prob_threshold)
        thr.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        # Selector de preprocesamiento (override manual)
        ttk.Label(top, text="Preprocess:").pack(side=tk.LEFT, padx=(6, 2))
        pp = ttk.Combobox(top, textvariable=self.preproc_choice, state="readonly",
                          values=["Auto", "ResNet", "MobileNetV3", "Genérico [0,1]"], width=16)
        pp.pack(side=tk.LEFT, padx=4)

        self.btn_start = ttk.Button(top, text="Iniciar cámara", command=self.start_camera, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_stop = ttk.Button(top, text="Detener cámara", command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        info = ttk.Frame(self.root, padding=8)
        info.pack(side=tk.TOP, fill=tk.X)

        self.lbl_model = ttk.Label(info, text="Modelo: –")
        self.lbl_model.pack(side=tk.LEFT, padx=6)

        self.lbl_shape = ttk.Label(info, text="Entrada: –")
        self.lbl_shape.pack(side=tk.LEFT, padx=6)

        ttk.Label(info, text="Arquitectura:").pack(side=tk.LEFT, padx=(12,2))
        ttk.Label(info, textvariable=self.arch_detected).pack(side=tk.LEFT)

        ttk.Label(info, text="FPS:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Label(info, textvariable=self.fps_value).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.root, width=900, height=560, bg="#111111")
        self.canvas.pack(padx=10, pady=10)

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bottom, text="Predicción:").pack(side=tk.LEFT)
        self.lbl_pred = ttk.Label(bottom, textvariable=self.pred_text, font=("Segoe UI", 12, "bold"))
        self.lbl_pred.pack(side=tk.LEFT, padx=10)

        foot = ttk.Frame(self.root, padding=8)
        foot.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(foot, text="Consejo: usa fondo uniforme y buena iluminación.").pack(side=tk.LEFT)

    # ------------------ Modelo/Labels ------------------
    def load_model_h5(self):
        path = filedialog.askopenfilename(
            title="Selecciona el modelo .h5",
            filetypes=[("Keras model", "*.h5"), ("Todos", "*.*")]
        )
        if not path:
            return
        try:
            self.model = load_model(path, compile=False)
            self.model_path = path
            self.input_shape = self._infer_input_shape()
            arch = detect_architecture(self.model)
            self.arch_detected.set(arch)
            self.lbl_model.configure(text=f"Modelo: {os.path.basename(path)}")
            self.lbl_shape.configure(text=f"Entrada: {self.input_shape}")
            self.pred_text.set("Modelo cargado. Puedes iniciar la cámara o cargar una imagen.")
            self.btn_start.configure(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error cargando modelo", str(e))
            self.model = None
            self.model_path = None
            self.input_shape = None
            self.btn_start.configure(state=tk.DISABLED)

    def load_labels(self):
        path = filedialog.askopenfilename(
            title="Selecciona labels (.txt o .json)",
            filetypes=[("Text or JSON", "*.txt *.json"), ("Todos", "*.*")]
        )
        if not path:
            return
        try:
            if path.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    self.labels = [line.strip() for line in f if line.strip()]
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        items = sorted(data.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
                        self.labels = [v for _, v in items]
                    elif isinstance(data, list):
                        self.labels = data
                    else:
                        raise ValueError("Formato JSON no soportado para labels.")
            messagebox.showinfo("Labels", f"{len(self.labels)} etiquetas cargadas.")
        except Exception as e:
            messagebox.showerror("Error cargando labels", str(e))
            self.labels = None

    def _infer_input_shape(self) -> Tuple[int, int, int]:
        try:
            in_shape = self.model.inputs[0].shape  # (None,H,W,C)
            h = int(in_shape[1]) if in_shape[1] is not None else 224
            w = int(in_shape[2]) if in_shape[2] is not None else 224
            c = int(in_shape[3]) if in_shape[3] is not None else 3
            return (h, w, c)
        except Exception:
            return (224, 224, 3)

    # ------------------ Cámara / Loop ------------------
    def start_camera(self):
        if self.model is None:
            messagebox.showwarning("Modelo no cargado", "Primero carga un modelo .h5")
            return
        idx = self.camera_index.get()
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            messagebox.showerror("Cámara", f"No se pudo abrir la cámara con índice {idx}")
            self.cap.release()
            self.cap = None
            return
        self.running = True
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.pred_text.set("Procesando…")
        self._video_loop()

    def stop_camera(self):
        self.running = False
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_start.configure(state=tk.NORMAL)
        self.fps_value.set("-")
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _video_loop(self):
        if not self.running or self.cap is None:
            return
        t0 = time.time()
        ret, frame = self.cap.read()
        if not ret:
            self.pred_text.set("No hay frame de la cámara")
            self.root.after(10, self._video_loop)
            return
        try:
            pred_label, pred_prob = self._predict_frame(frame)
            thr = self.prob_threshold.get()
            if pred_label is None:
                self.pred_text.set("–")
            elif pred_prob >= thr:
                self.pred_text.set(f"{pred_label}  ({pred_prob:.2f})")
            else:
                self.pred_text.set(f"(<{thr:.2f})  {pred_label}  ({pred_prob:.2f})")
        except Exception as e:
            self.pred_text.set(f"Error de predicción: {e}")

        vis = self._draw_overlay(frame, self.pred_text.get())
        self._show_on_canvas(vis)

        dt = max(time.time() - t0, 1e-6)
        self.fps_value.set(f"{1.0/dt:.1f}")
        self.root.after(1, self._video_loop)

    # ------------------ Predicción ------------------
    def _effective_preprocess_mode(self) -> str:
        choice = self.preproc_choice.get()
        if choice == "ResNet": return "resnet"
        if choice == "MobileNetV3": return "mobilenet_v3"
        if choice == "Genérico [0,1]": return "generic"
        # Auto:
        return self.arch_detected.get() if self.arch_detected.get() in ("resnet","mobilenet_v3") else "generic"

    def _predict_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        H, W, C = self.input_shape if self.input_shape else (224, 224, 3)
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if C == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img.astype(np.float32), axis=-1)  # H,W,1
        else:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)  # todavía 0..255

        # Preprocess correcto
        mode = self._effective_preprocess_mode()
        x = np.expand_dims(preprocess_for(mode, img), axis=0)  # 1,H,W,C

        preds = self.model.predict(x, verbose=0)

        # Categórico estándar: (1, num_clases)
        if preds.ndim == 2 and preds.shape[0] == 1 and preds.shape[1] > 1:
            probs = preds[0]
            # Si no parecen probs (suman!=1 o fuera de [0,1]), aplica softmax
            s = float(np.sum(probs))
            if (np.min(probs) < 0.0) or (np.max(probs) > 1.0) or (abs(s-1.0) > 1e-3):
                ex = np.exp(probs - np.max(probs))
                probs = ex / np.sum(ex)
            idx = int(np.argmax(probs))
            prob = float(probs[idx])
            label = self._idx_to_label(idx)
            return label, prob

        # Binario (1,1) con sigmoid
        if preds.ndim == 2 and preds.shape[1] == 1:
            p = float(preds[0,0])
            label = self._idx_to_label(1 if p >= 0.5 else 0)
            return label, p if p >= 0.5 else 1.0 - p

        # Caso raro
        val = float(np.ravel(preds)[0])
        return f"valor: {val:.3f}", 1.0

    def _idx_to_label(self, idx: int) -> str:
        if self.labels and 0 <= idx < len(self.labels):
            return self.labels[idx]
        return f"clase_{idx}"

    # ------------------ Visualización ------------------
    def _draw_overlay(self, frame_bgr: np.ndarray, text: str) -> np.ndarray:
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
        cv2.putText(vis, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    def _show_on_canvas(self, img_rgb: np.ndarray):
        ih, iw = img_rgb.shape[:2]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        scale = min(cw/iw, ch/ih)
        nw, nh = int(iw*scale), int(ih*scale)
        resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(resized)
        self.tk_img = ImageTk.PhotoImage(image=image)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self.tk_img)

    # ------------------ Imagen estática ------------------
    def load_image_and_predict(self):
        if self.model is None:
            messagebox.showwarning("Modelo no cargado", "Primero carga un modelo .h5")
            return
        if self.running:
            self.stop_camera()

        path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Imagen", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Todos", "*.*")]
        )
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Imagen", f"No se pudo cargar: {os.path.basename(path)}")
            return

        try:
            label, prob = self._predict_frame(img_bgr)
            text = f"{label} ({prob:.2f})"
            self.pred_text.set(text)
            vis = self._draw_overlay(img_bgr, text)
            self._show_on_canvas(vis)
        except Exception as e:
            messagebox.showerror("Predicción", f"Error: {e}")

    # ------------------ Cleanup ------------------
    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = RealtimeSignApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
