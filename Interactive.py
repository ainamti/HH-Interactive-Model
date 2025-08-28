import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import math


class HodgkinHuxleyInteractive:
    def __init__(self):
        # Initialize parameters (matching your original code)
        self.dt = 10e-6
        self.Cm = 100e-12
        self.v_init = -80e-3
        self.Gbar_k = 1e-6
        self.Gbar_na = 7e-6
        self.Gleak = 5e-9
        self.Ek = -80e-3
        self.Ena = 40e-3
        self.Eleak = -70e-3
        self.current_magnitude = 200e-12
        self.current_start = 0.2
        self.current_duration = 0.3
        self.total_time = 1.0

        # Setup the figure and subplots
        self.fig = plt.figure(figsize=(15, 10))

        # Create subplot layout
        self.ax_voltage = plt.subplot(3, 2, (1, 2))  # Top row, spans 2 columns
        self.ax_gates = plt.subplot(3, 2, 3)  # Middle left
        self.ax_currents = plt.subplot(3, 2, 4)  # Middle right

        # Space for sliders at bottom
        plt.subplots_adjust(bottom=0.35)

        # Initialize plot lines
        (self.line_voltage,) = self.ax_voltage.plot(
            [], [], "b-", linewidth=2, label="Membrane Potential"
        )
        (self.line_stimulus,) = self.ax_voltage.plot(
            [], [], "r-", linewidth=2, label="Stimulus"
        )

        (self.line_m,) = self.ax_gates.plot(
            [], [], "r-", linewidth=2, label="m (Na+ activation)"
        )
        (self.line_h,) = self.ax_gates.plot(
            [], [], "orange", linewidth=2, label="h (Na+ inactivation)"
        )
        (self.line_n,) = self.ax_gates.plot(
            [], [], "g-", linewidth=2, label="n (K+ activation)"
        )

        (self.line_gna,) = self.ax_currents.plot(
            [], [], "r-", linewidth=2, label="Na+ Conductance"
        )
        (self.line_gk,) = self.ax_currents.plot(
            [], [], "g-", linewidth=2, label="K+ Conductance"
        )

        # Setup axes
        self.setup_axes()

        # Create sliders
        self.create_sliders()

        # Run initial simulation
        self.update_simulation()

    def alpha_n(self, v):
        v = v * 1000
        return 0.01 * (-v - 55) / (math.exp((-v - 55) / 10.0) - 1) * 1000

    def beta_n(self, v):
        v = v * 1000
        return 0.125 * math.exp((-v - 65) / 80.0) * 1000

    def alpha_m(self, v):
        v = v * 1000
        return 0.1 * (-v - 40) / (math.exp((-v - 40) / 10.0) - 1) * 1000

    def beta_m(self, v):
        v = v * 1000
        return 4 * math.exp((-v - 65) / 18.0) * 1000

    def alpha_h(self, v):
        v = v * 1000
        return 0.07 * math.exp((-v - 65) / 20.0) * 1000

    def beta_h(self, v):
        v = v * 1000
        return 1 / (math.exp((-v - 35) / 10.0) + 1) * 1000

    def setup_axes(self):
        # Voltage plot
        self.ax_voltage.set_xlabel("Time (s)")
        self.ax_voltage.set_ylabel("Voltage (V)")
        self.ax_voltage.set_title("Membrane Potential")
        self.ax_voltage.legend()
        self.ax_voltage.grid(True)

        # Gates plot
        self.ax_gates.set_xlabel("Time (s)")
        self.ax_gates.set_ylabel("Gate Variable")
        self.ax_gates.set_title("Channel Gate Variables")
        self.ax_gates.legend()
        self.ax_gates.grid(True)
        self.ax_gates.set_ylim(0, 1)

        # Currents plot
        self.ax_currents.set_xlabel("Time (s)")
        self.ax_currents.set_ylabel("Conductance (S)")
        self.ax_currents.set_title("Channel Conductances")
        self.ax_currents.legend()
        self.ax_currents.grid(True)

    def create_sliders(self):
        # Define slider positions
        slider_height = 0.03
        slider_spacing = 0.04
        left_margin = 0.1
        right_margin = 0.5
        width = 0.35

        # Create sliders
        self.sliders = {}

        # Left column sliders
        y_positions_left = [0.25, 0.20, 0.15, 0.10, 0.05]
        slider_params_left = [
            (
                "Gbar_na",
                "Na+ Conductance (µS)",
                self.Gbar_na * 1e6,
                2,
                15,
                lambda x: x / 1e6,
            ),
            (
                "Gbar_k",
                "K+ Conductance (µS)",
                self.Gbar_k * 1e6,
                0.5,
                3,
                lambda x: x / 1e6,
            ),
            (
                "Gleak",
                "Leak Conductance (nS)",
                self.Gleak * 1e9,
                1,
                20,
                lambda x: x / 1e9,
            ),
            ("Cm", "Capacitance (pF)", self.Cm * 1e12, 50, 200, lambda x: x / 1e12),
            (
                "current_magnitude",
                "Current (pA)",
                self.current_magnitude * 1e12,
                0,
                500,
                lambda x: x / 1e12,
            ),
        ]

        for i, (param, label, initial, min_val, max_val, converter) in enumerate(
            slider_params_left
        ):
            ax_slider = plt.axes(
                [left_margin, y_positions_left[i], width, slider_height]
            )
            slider = Slider(ax_slider, label, min_val, max_val, valinit=initial)
            slider.converter = converter
            slider.on_changed(self.update_parameter)
            self.sliders[param] = slider

        # Right column sliders
        y_positions_right = [0.25, 0.20, 0.15]
        slider_params_right = [
            (
                "current_start",
                "Start Time (ms)",
                self.current_start * 1000,
                100,
                500,
                lambda x: x / 1000,
            ),
            (
                "current_duration",
                "Duration (ms)",
                self.current_duration * 1000,
                50,
                500,
                lambda x: x / 1000,
            ),
            ("total_time", "Total Time (s)", self.total_time, 0.5, 2.0, lambda x: x),
        ]

        for i, (param, label, initial, min_val, max_val, converter) in enumerate(
            slider_params_right
        ):
            ax_slider = plt.axes(
                [right_margin, y_positions_right[i], width, slider_height]
            )
            slider = Slider(ax_slider, label, min_val, max_val, valinit=initial)
            slider.converter = converter
            slider.on_changed(self.update_parameter)
            self.sliders[param] = slider

        # Add reset button
        ax_reset = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.reset_button = Button(ax_reset, "Reset")
        self.reset_button.on_clicked(self.reset_parameters)

    def update_parameter(self, val):
        # Find which slider was changed and update the corresponding parameter
        for param_name, slider in self.sliders.items():
            if hasattr(slider, "converter"):
                setattr(self, param_name, slider.converter(slider.val))
        self.update_simulation()

    def reset_parameters(self, event):
        # Reset all parameters to defaults
        self.Gbar_k = 1e-6
        self.Gbar_na = 7e-6
        self.Gleak = 5e-9
        self.Cm = 100e-12
        self.current_magnitude = 200e-12
        self.current_start = 0.2
        self.current_duration = 0.3
        self.total_time = 1.0

        # Reset sliders
        self.sliders["Gbar_na"].reset()
        self.sliders["Gbar_k"].reset()
        self.sliders["Gleak"].reset()
        self.sliders["Cm"].reset()
        self.sliders["current_magnitude"].reset()
        self.sliders["current_start"].reset()
        self.sliders["current_duration"].reset()
        self.sliders["total_time"].reset()

        self.update_simulation()

    def run_simulation(self):
        # Calculate simulation parameters
        total_steps = int(round(self.total_time / self.dt))
        current_start_step = int(round(self.current_start / self.dt))
        current_end_step = int(
            round((self.current_start + self.current_duration) / self.dt)
        )

        # Initialize arrays
        v_out = np.zeros(total_steps)
        m_out = np.zeros(total_steps)
        h_out = np.zeros(total_steps)
        n_out = np.zeros(total_steps)
        gna_out = np.zeros(total_steps)
        gk_out = np.zeros(total_steps)
        stimulus = np.zeros(total_steps)

        # Initial conditions
        n = self.alpha_n(self.v_init) / (
            self.alpha_n(self.v_init) + self.beta_n(self.v_init)
        )
        m = self.alpha_m(self.v_init) / (
            self.alpha_m(self.v_init) + self.beta_m(self.v_init)
        )
        h = self.alpha_h(self.v_init) / (
            self.alpha_h(self.v_init) + self.beta_h(self.v_init)
        )

        # Time loop (matching your original code structure)
        for t in range(total_steps):
            if t == 0:
                v_out[t] = self.v_init
            else:
                # Calculate particle changes
                dn = (
                    self.alpha_n(v_out[t - 1]) * (1 - n) - self.beta_n(v_out[t - 1]) * n
                ) * self.dt
                n = n + dn

                dm = (
                    self.alpha_m(v_out[t - 1]) * (1 - m) - self.beta_m(v_out[t - 1]) * m
                ) * self.dt
                m = m + dm

                dh = (
                    self.alpha_h(v_out[t - 1]) * (1 - h) - self.beta_h(v_out[t - 1]) * h
                ) * self.dt
                h = h + dh

                # Calculate conductances
                Gk = self.Gbar_k * n**4
                Gna = self.Gbar_na * m**3 * h

                # Calculate currents
                i_k = Gk * (v_out[t - 1] - self.Ek)
                i_na = Gna * (v_out[t - 1] - self.Ena)
                i_leak = self.Gleak * (v_out[t - 1] - self.Eleak)

                # Injected current
                i_inj = (
                    self.current_magnitude
                    if current_start_step <= t < current_end_step
                    else 0
                )

                # Sum currents
                i_cap = i_inj - i_leak - i_k - i_na

                # Calculate voltage change
                dv = (i_cap / self.Cm) * self.dt
                v_out[t] = v_out[t - 1] + dv

                # Store conductances
                gna_out[t] = Gna
                gk_out[t] = Gk

            # Store gate variables and stimulus
            m_out[t] = m
            h_out[t] = h
            n_out[t] = n
            stimulus[t] = (
                self.current_magnitude
                if current_start_step <= t < current_end_step
                else 0
            )

        # Create time vector
        t_vec = np.linspace(0, self.dt * total_steps, total_steps)

        return t_vec, v_out, m_out, h_out, n_out, gna_out, gk_out, stimulus

    def update_simulation(self):
        # Run simulation
        t_vec, v_out, m_out, h_out, n_out, gna_out, gk_out, stimulus = (
            self.run_simulation()
        )

        # Update voltage plot
        self.line_voltage.set_data(t_vec, v_out)
        self.line_stimulus.set_data(
            t_vec, stimulus * 1000
        )  # Scale stimulus for visibility

        # Update gates plot
        self.line_m.set_data(t_vec, m_out)
        self.line_h.set_data(t_vec, h_out)
        self.line_n.set_data(t_vec, n_out)

        # Update conductances plot
        self.line_gna.set_data(t_vec, gna_out)
        self.line_gk.set_data(t_vec, gk_out)

        # Update axis limits
        self.ax_voltage.set_xlim(0, max(t_vec))
        self.ax_voltage.set_ylim(min(v_out) * 1.1, max(v_out) * 1.1)
        self.ax_voltage.relim()
        self.ax_voltage.autoscale_view()

        self.ax_gates.set_xlim(0, max(t_vec))
        self.ax_gates.relim()
        self.ax_gates.autoscale_view()

        self.ax_currents.set_xlim(0, max(t_vec))
        self.ax_currents.set_ylim(0, max(max(gna_out), max(gk_out)) * 1.1)
        self.ax_currents.relim()
        self.ax_currents.autoscale_view()

        # Refresh the plot
        self.fig.canvas.draw()

    def show(self):
        plt.show()


# Run the interactive model
if __name__ == "__main__":
    print("Starting Interactive Hodgkin-Huxley Model...")
    print("Adjust the sliders to see real-time changes in the simulation!")
    print("Close the plot window to exit.")

    model = HodgkinHuxleyInteractive()
    model.show()
