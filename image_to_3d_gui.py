import cv2
import numpy as np
from stl import mesh
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import vtk


class ImageTo3D:
    def __init__(self, master):
        self.master = master
        self.master.title("2D Image to 3D Model Converter")
        self.master.geometry("400x600")
        self.master.configure(bg='#f0f0f0')

        # Initialize variables
        self.image_path = None
        self.height_map = None
        self.scale = 10  # Scale factor (unitless; height units will be in the same unit system)
        self.threshold = 0  # Height threshold (grayscale value; between 0-255)
        self.vertices = None
        self.faces = None

        # Set default values for scale and threshold
        self.default_scale = 10  # Default scale factor
        self.default_threshold = 0  # Default height threshold

        # GUI Elements
        self.create_widgets()

    def create_widgets(self):
        # Upload Image Button
        self.upload_btn = tk.Button(self.master, text="Upload Image", command=self.upload_image, bg='#007BFF', fg='white')
        self.upload_btn.pack(pady=10)

        # Scale Entry
        self.scale_label = tk.Label(self.master, text="Scale Factor (unitless):", bg='#f0f0f0')
        self.scale_label.pack()
        self.scale_entry = tk.Entry(self.master)
        self.scale_entry.insert(0, str(self.default_scale))  # Set default scale factor
        self.scale_entry.pack()

        # Height Threshold Entry
        self.threshold_label = tk.Label(self.master, text="Height Threshold (0-255):", bg='#f0f0f0')
        self.threshold_label.pack()
        self.threshold_entry = tk.Entry(self.master)
        self.threshold_entry.insert(0, str(self.default_threshold))  # Set default threshold value
        self.threshold_entry.pack()

        # Smoothing Options
        self.smooth_var = tk.StringVar(value="None")
        self.smooth_label = tk.Label(self.master, text="Select Smoothing:", bg='#f0f0f0')
        self.smooth_label.pack()
        self.smooth_none = tk.Radiobutton(self.master, text="None", variable=self.smooth_var, value="None", bg='#f0f0f0')
        self.smooth_none.pack()
        self.smooth_gaussian = tk.Radiobutton(self.master, text="Gaussian", variable=self.smooth_var, value="Gaussian", bg='#f0f0f0')
        self.smooth_gaussian.pack()
        self.smooth_median = tk.Radiobutton(self.master, text="Median", variable=self.smooth_var, value="Median", bg='#f0f0f0')
        self.smooth_median.pack()

        # Generate Model Button
        self.generate_btn = tk.Button(self.master, text="Generate 3D Model", command=self.generate_model, bg='#28A745', fg='white')
        self.generate_btn.pack(pady=10)

        # Visualization Button
        self.visualize_btn = tk.Button(self.master, text="Visualize Height Map", command=self.visualize_height_map, bg='#17A2B8', fg='white')
        self.visualize_btn.pack(pady=10)

        # Save STL Button
        self.save_btn = tk.Button(self.master, text="Save as STL", command=self.save_model, bg='#FFC107', fg='black')
        self.save_btn.pack(pady=10)

        # Show 3D Model Button
        self.show_model_btn = tk.Button(self.master, text="Show 3D Model", command=self.show_3d_model, bg='#6F42C1', fg='white')
        self.show_model_btn.pack(pady=10)

        # Status Label
        self.status_label = tk.Label(self.master, text="", fg="blue", bg='#f0f0f0')
        self.status_label.pack(pady=10)

        # Image Display Label
        self.image_label = tk.Label(self.master, bg='#f0f0f0')
        self.image_label.pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self.master, length=200, mode='determinate')
        self.progress.pack(pady=10)

    def update_status(self, message):
        """Update the status message in the GUI."""
        self.status_label.config(text=message)
        self.master.update_idletasks()

    def upload_image(self):
        """Upload image and convert it to a height map."""
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not self.image_path:
            return

        self.update_status("Loading image...")
        image = cv2.imread(self.image_path)
        if image is None:
            messagebox.showerror("Error", "Could not open image.")
            return

        # Convert to grayscale
        self.height_map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.update_status("Image loaded. Use the settings to generate the 3D model.")

        # Display the loaded image
        self.display_image(image)

    def apply_smoothing(self):
        """Apply selected smoothing to the height map."""
        self.update_status("Applying smoothing...")
        if self.smooth_var.get() == "Gaussian":
            self.height_map = cv2.GaussianBlur(self.height_map, (5, 5), 0)
        elif self.smooth_var.get() == "Median":
            self.height_map = cv2.medianBlur(self.height_map, 5)
        self.update_status("Smoothing applied.")

    def generate_model(self):
        """Generate the 3D model from the height map."""
        if self.height_map is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Retrieve scale factor and height threshold
        self.scale = float(self.scale_entry.get())  # Scale factor (unitless)
        self.threshold = float(self.threshold_entry.get())  # Height threshold (grayscale value)

        self.apply_smoothing()

        # Create height map
        normalized_height_map = self.height_map / 255.0
        normalized_height_map[normalized_height_map < self.threshold / 255.0] = 0  # Apply height threshold

        # Update status and start model generation
        self.update_status("Generating 3D model...")
        self.progress.start()

        # Generate the 3D mesh
        self.vertices, self.faces = self.create_3d_mesh(normalized_height_map)

        self.progress.stop()
        self.update_status("3D model generated successfully!")

    def create_3d_mesh(self, height_map):
        """Generate a 3D mesh from the height map."""
        rows, cols = height_map.shape
        vertices = []
        faces = []

        for i in range(rows - 1):
            for j in range(cols - 1):
                # Create vertices for each square in the height map
                v0 = (j, height_map[i, j] * self.scale, i)
                v1 = (j + 1, height_map[i, j + 1] * self.scale, i)
                v2 = (j, height_map[i + 1, j] * self.scale, i + 1)
                v3 = (j + 1, height_map[i + 1, j + 1] * self.scale, i + 1)

                # Append vertices
                idx = len(vertices)
                vertices.extend([v0, v1, v2, v3])

                # Create two faces (triangles) for each square
                faces.append([idx, idx + 1, idx + 2])
                faces.append([idx + 1, idx + 3, idx + 2])

        return np.array(vertices), np.array(faces)

    def visualize_height_map(self):
        """Visualize the height map using matplotlib."""
        if self.height_map is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        plt.imshow(self.height_map, cmap='gray')
        plt.colorbar()
        plt.title("Height Map")
        plt.show()

    def save_model(self):
        """Save the generated model as an STL file."""
        if self.vertices is None or self.faces is None:
            messagebox.showerror("Error", "Please generate a model first.")
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".stl",
                                                     filetypes=[("STL Files", "*.stl")])
        if not output_path:
            return

        # Create the mesh
        model_mesh = mesh.Mesh(np.zeros(self.faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(self.faces):
            for j in range(3):
                model_mesh.vectors[i][j] = self.vertices[f[j]]

        model_mesh.save(output_path)
        messagebox.showinfo("Success", "Model saved as STL successfully!")

    def show_3d_model(self):
        """Display the generated 3D model using VTK."""
        if self.vertices is None or self.faces is None:
            messagebox.showerror("Error", "Please generate a model first.")
            return

        # Create a VTK polydata object
        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        # Add points to VTK
        for vertex in self.vertices:
            points.InsertNextPoint(vertex)

        # Add faces to VTK
        for face in self.faces:
            cells.InsertNextCell(3, face)

        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create a renderer, render window, and interactor
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        # Add the actor to the scene
        renderer.AddActor(actor)
        renderer.SetBackground(1, 1, 1)  # Background color

        # Start the visualization
        render_window.Render()
        render_window_interactor.Start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageTo3D(root)
    root.mainloop()
