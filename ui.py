# ui.py

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import mesh

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class MeshUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Polygon Mesh Tool")

        # STATE
        self.image_path = None
        self.image = None
        self.img_tk = None
        self.polygon = None
        self.tri_vertices = None
        self.tri_faces = None
        self.mode = tk.StringVar(value="View")
        self.info_text = tk.StringVar(value="Ready.")
        self.history = []  # For undoing refinements

        # Circle selector state (for Refine mode)
        self.circle_center = np.array([100, 100], dtype=float)
        self.circle_radius = 40.0
        self.circle_drag = False
        self.circle_resize = False
        self._last_mouse = None

        # Refine points setting
        self.refine_points = tk.IntVar(value=2)

        # Mesh density spacing slider variable
        self.mesh_spacing = tk.IntVar(value=10)

        # ============= UI LAYOUT =============
        # Main display (Matplotlib)
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self.on_canvas_release)
        self.canvas.mpl_connect("scroll_event", self.on_canvas_scroll)

        # Sidebar (Frame)
        sidebar = tk.Frame(master, width=220, bg="#f8f8f8")
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Top: File ops ---
        file_frame = tk.LabelFrame(sidebar, text="File", padx=5, pady=5)
        file_frame.pack(fill=tk.X, padx=6, pady=8)
        tk.Button(file_frame, text="Import PNG", command=self.load_png).pack(fill=tk.X, pady=1)
        tk.Button(file_frame, text="Export Polygon", command=self.export_polygon).pack(fill=tk.X, pady=1)
        tk.Button(file_frame, text="Export Mesh", command=self.export_mesh).pack(fill=tk.X, pady=1)

        # --- Middle: Mode selection ---
        mode_frame = tk.LabelFrame(sidebar, text="Mode", padx=5, pady=5)
        mode_frame.pack(fill=tk.X, padx=6, pady=8)
        for mode in ("View", "Extract", "Refine", "Mesh"):
            tk.Radiobutton(mode_frame, text=mode, variable=self.mode, value=mode, command=self.on_mode_change).pack(anchor='w', pady=1)

        # --- Actions for active mode ---
        self.action_frame = tk.LabelFrame(sidebar, text="Actions", padx=5, pady=5)
        self.action_frame.pack(fill=tk.X, padx=6, pady=8)

        # Undo button
        self.btn_undo = tk.Button(sidebar, text="Undo", command=self.undo, state=tk.DISABLED)
        self.btn_undo.pack(fill=tk.X, padx=6, pady=4)

        # --- Info panel ---
        self.info_panel = tk.Label(sidebar, textvariable=self.info_text, bg="#f8f8f8", wraplength=200, justify=tk.LEFT)
        self.info_panel.pack(fill=tk.X, padx=6, pady=10, anchor='s')

        # Draw initial state
        self.update_action_panel()
        self.update_plot()

    # ---------- Mode Switching ----------
    def on_mode_change(self):
        self.update_action_panel()
        self.update_plot()

    def update_action_panel(self):
        for child in self.action_frame.winfo_children():
            child.destroy()
        mode = self.mode.get()
        if mode == "Extract":
            tk.Button(self.action_frame, text="Extract Outline", command=self.extract_outline).pack(fill=tk.X)
        elif mode == "Refine":
            tk.Button(self.action_frame, text="Apply Circle Refine", command=self.apply_circle_refine).pack(fill=tk.X, pady=(0,2))
            tk.Label(self.action_frame, text="Points per segment:", fg="gray").pack(pady=(4,0))
            refine_slider = tk.Scale(
                self.action_frame, from_=2, to=5, orient=tk.HORIZONTAL, variable=self.refine_points,
                showvalue=True, resolution=1, command=lambda e: self.update_plot())
            refine_slider.pack(fill=tk.X)
            tk.Label(self.action_frame, text="Drag center or scroll to resize.\nRed dots preview new points.", fg="gray").pack()
        elif mode == "Mesh":
            tk.Label(self.action_frame, text="Interior Point Spacing:").pack()
            mesh_slider = tk.Scale(
                self.action_frame, from_=4, to=40, orient=tk.HORIZONTAL,
                variable=self.mesh_spacing, showvalue=True, resolution=1
            )
            mesh_slider.pack(fill=tk.X)
            tk.Button(self.action_frame, text="Triangulate", command=self.generate_mesh).pack(fill=tk.X)

    # ---------- File I/O ----------
    def load_png(self):
        path = filedialog.askopenfilename(filetypes=[("PNG Images", "*.png")])
        if not path: return
        self.image_path = path
        self.image = Image.open(path).convert("RGBA")
        img = self.image.copy()
        img.thumbnail((400, 400))
        self.img_tk = ImageTk.PhotoImage(img)
        self.polygon = None
        self.tri_vertices = None
        self.tri_faces = None
        self.info_text.set(f"Loaded: {path.split('/')[-1]}")
        self.history.clear()
        self.btn_undo.config(state=tk.DISABLED)
        self.update_plot()

    def export_polygon(self):
        if self.polygon is None:
            messagebox.showinfo("Export Polygon", "No polygon to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("TXT", "*.txt")])
        if not path: return
        mesh.export_polygon_as_txt(self.polygon, path)
        self.info_text.set(f"Polygon exported to {path}")

    def export_mesh(self):
        if self.tri_vertices is None or self.tri_faces is None:
            messagebox.showinfo("Export Mesh", "No mesh to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension="_mesh", filetypes=[("TXT", "*.txt")])
        if not path: return
        mesh.export_mesh_as_txt(self.tri_vertices, self.tri_faces, path)
        self.info_text.set(f"Mesh exported to {path}_vertices.txt and {path}_faces.txt")

    # ---------- Undo logic ----------
    def undo(self):
        if self.history:
            self.polygon = self.history.pop()
            self.info_text.set("Undo successful.")
            self.update_plot()
            if not self.history:
                self.btn_undo.config(state=tk.DISABLED)
        else:
            self.info_text.set("Nothing to undo.")

    # ---------- Polygon extraction and refinement ----------
    def extract_outline(self):
        if not self.image_path:
            self.info_text.set("Load a PNG first.")
            return
        try:
            poly = mesh.load_and_extract_polygon(self.image_path, simplify=0.003)
            self.polygon = poly
            self.tri_vertices = None
            self.tri_faces = None
            self.info_text.set(f"Extracted outline ({len(poly)} points).")
            self.history.clear()
            self.btn_undo.config(state=tk.DISABLED)
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract polygon: {e}")

    def apply_circle_refine(self):
        if self.polygon is None:
            self.info_text.set("Extract outline first.")
            return
        # Save to history for undo
        self.history.append(self.polygon.copy())
        self.btn_undo.config(state=tk.NORMAL)
        num_pts = self.refine_points.get()
        new_poly = mesh.refine_polygon_with_circle(self.polygon, self.circle_center, self.circle_radius, points_per_segment=num_pts)
        self.polygon = new_poly
        self.info_text.set(f"Refined: {len(new_poly)} points. You can refine again or switch modes.")
        self.update_plot()

    def generate_mesh(self):
        if self.polygon is None:
            self.info_text.set("Extract or load a polygon first.")
            return
        try:
            spacing = self.mesh_spacing.get()
            self.tri_vertices, self.tri_faces = mesh.get_triangulation(self.polygon, spacing=spacing)
            self.info_text.set(f"Triangulated ({len(self.tri_faces)} faces).")
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate mesh: {e}")

    # ---------- Drawing & Canvas logic ----------
    def update_plot(self):
        self.ax.clear()
        mode = self.mode.get()

        if self.image is not None and (mode == "View" or (mode == "Extract" and self.polygon is None)):
            img = self.image.copy()
            img.thumbnail((400, 400))
            self.ax.imshow(img)
            self.ax.set_title("Image")
        if self.polygon is not None:
            poly = self.polygon
            self.ax.fill(poly[:,0], poly[:,1], color='orange', alpha=0.2)
            self.ax.plot(poly[:,0], poly[:,1], color='orange')
            for i, (x, y) in enumerate(poly):
                self.ax.text(x, y, str(i), fontsize=8, color='blue')

        if mode == "Refine" and self.polygon is not None:
            # Draw selector circle
            self.circle_selector = mpatches.Circle(self.circle_center, self.circle_radius,
                                                   fill=False, color='red', lw=2)
            self.ax.add_patch(self.circle_selector)
            # Center drag handle
            handle_radius = max(4, self.circle_radius * 0.08)
            handle = mpatches.Circle(self.circle_center, handle_radius, color='red', alpha=0.6)
            self.ax.add_patch(handle)
            # Live preview of all new points
            num_pts = self.refine_points.get()
            preview_pts = mesh.preview_circle_polygon_intersections(self.polygon, self.circle_center, self.circle_radius, points_per_segment=num_pts)
            if preview_pts is not None and len(preview_pts) > 0:
                self.ax.scatter(preview_pts[:,0], preview_pts[:,1], color='red', zorder=10, label="Intersections")
            self.ax.set_title(f"Circle Refine: {num_pts} pts/segment")

        if mode == "Mesh" and self.tri_vertices is not None and self.tri_faces is not None:
            verts, faces = self.tri_vertices, self.tri_faces
            self.ax.triplot(verts[:,0], verts[:,1], faces, color='purple', lw=1)
            self.ax.set_title("Mesh (CDT)")

        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.fig.tight_layout()
        self.fig.canvas.draw()

    # ---------- Canvas interaction ----------
    def on_canvas_click(self, event):
        if self.mode.get() != "Refine" or self.polygon is None or event.inaxes != self.ax:
            return
        mouse = np.array([event.xdata, event.ydata], dtype=float)
        dist_to_center = np.linalg.norm(mouse - self.circle_center)
        handle_radius = max(4, self.circle_radius * 0.08)
        # Resize: near edge, otherwise drag: near center
        if abs(dist_to_center - self.circle_radius) < max(5, self.circle_radius*0.12):
            self.circle_resize = True
            self._last_mouse = mouse
        elif dist_to_center < handle_radius:
            self.circle_drag = True
            self._last_mouse = mouse
        else:
            self.circle_drag = False
            self.circle_resize = False

    def on_canvas_motion(self, event):
        if self.mode.get() != "Refine" or self.polygon is None or event.inaxes != self.ax:
            return
        mouse = np.array([event.xdata, event.ydata], dtype=float)
        if self.circle_drag and self._last_mouse is not None:
            delta = mouse - self._last_mouse
            self.circle_center += delta
            self._last_mouse = mouse
            self.update_plot()
        elif self.circle_resize and self._last_mouse is not None:
            self.circle_radius = np.linalg.norm(mouse - self.circle_center)
            self._last_mouse = mouse
            self.update_plot()

    def on_canvas_release(self, event):
        self.circle_drag = False
        self.circle_resize = False
        self._last_mouse = None

    def on_canvas_scroll(self, event):
        if self.mode.get() != "Refine" or self.polygon is None or event.inaxes != self.ax:
            return
        # Scrolling up increases radius, down decreases
        factor = 1.1 if event.button == 'up' else 0.9
        self.circle_radius *= factor
        self.update_plot()

if __name__ == '__main__':
    root = tk.Tk()
    app = MeshUI(root)
    root.mainloop()
