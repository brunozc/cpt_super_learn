import os
import sys
import random
import numpy as np
import pygame
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid conflicts with Pygame
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from CPTSuperLearn.environment import CPTEnvironment
from CPTSuperLearn.agent import DQLAgent
from CPTSuperLearn.utils import read_data_file


class CPTGame:
    """Interactive digital twin for CPT sampling strategy"""

    def __init__(self, validation_data_folder, model_path, screen_width=1200, screen_height=800):
        """
        Initialize the CPT game

        Parameters:
        -----------
        validation_data_folder: folder with validation data
        model_path: path to the trained model
        screen_width: width of the game window
        screen_height: height of the game window
        """
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("CPT Super Learn - Digital Twin")
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()

        # Game dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.field_width = screen_width * 0.8
        self.field_height = screen_height * 0.25

        # Start screen state
        self.in_start_screen = True

        # Load or create truck image
        try:
            truck_path = os.path.join(os.path.dirname(__file__), "assets", "cpt_truck.png")
            if os.path.exists(truck_path):
                self.truck_img = pygame.image.load(truck_path)
            else:
                # Create a simple truck if image doesn't exist
                self.truck_img = self._create_simple_truck(50, 30)
        except:
            # Fallback to simple truck if there's any loading issue
            self.truck_img = self._create_simple_truck(50, 30)

        self.truck_img = pygame.transform.scale(self.truck_img, (50, 30))

        # Load environment and agent
        self.cpt_env = CPTEnvironment.load_environment(model_path)
        self.agent = DQLAgent.load_model(model_path)

        # Load validation data
        self.validation_folder = validation_data_folder
        self.validation_files = [f for f in os.listdir(validation_data_folder) if f.endswith('.csv')]

        # Game state
        self.current_file = None
        self.current_data = None
        self.auto_mode = False
        self.game_over = False
        self.step_count = 0
        self.max_steps = self.cpt_env.max_nb_cpts

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.LIGHT_BLUE = (100, 180, 255)
        self.DARK_BLUE = (0, 80, 200)

        # Fonts
        self.font = pygame.font.SysFont('Arial', 24)
        self.title_font = pygame.font.SysFont('Arial', 48, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 16)

        # UI elements
        self.cursor_position = 0

        # Start button
        self.start_button = {
            'rect': pygame.Rect(self.screen_width//2 - 100, self.screen_height - 150, 200, 60),
            'color': self.LIGHT_BLUE,
            'hover_color': self.DARK_BLUE,
            'text': 'START GAME',
            'hovered': False
        }

        # Plot area dimensions (these will be calculated when rendering)
        self.plot_left_margin = 20
        self.plot_width = int(self.field_width)
        self.plot_actual_width = None  # Will be set based on the actual data
        self.plot_start_x = None       # Will be set after rendering

    def _create_simple_truck(self, width, height):
        """Create a simple truck image"""
        truck_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Draw truck body (rectangle)
        pygame.draw.rect(truck_surface, (50, 50, 150), (10, 5, width-20, height-15))

        # Draw cab (rectangle)
        pygame.draw.rect(truck_surface, (50, 50, 150), (width-15, 10, 10, height-20))

        # Draw wheels (circles)
        pygame.draw.circle(truck_surface, (0, 0, 0), (15, height-5), 5)
        pygame.draw.circle(truck_surface, (0, 0, 0), (width-15, height-5), 5)

        return truck_surface

    def load_random_file(self):
        """Load a random file from the validation dataset"""
        random_file = random.choice(self.validation_files)
        file_name, image_data = read_data_file(os.path.join(self.validation_folder, random_file))
        self.current_file = file_name
        self.current_data = image_data

        # Reset environment with the new file
        state = self.cpt_env.reset(file_name, image_data)
        self.step_count = 1  # We already have one CPT from reset
        self.game_over = False
        return state

    def manual_step(self, position):
        """Take a manual step in the environment"""
        # Use the exact position passed in - this will be cursor_position
        current_pos = self.cpt_env.sampled_positions[-1]
        distance = position - current_pos

        # Find the closest action - this is unchanged
        action_distances = [abs(act - distance) for act in self.cpt_env.action_list]
        action_index = action_distances.index(min(action_distances))

        return self.take_step(action_index)

    def auto_step(self, state):
        """Let the AI select the next step"""
        action_index = self.agent.get_next_action(state, training=False)
        return self.take_step(action_index)

    def take_step(self, action_index):
        """Take a step with the given action index"""
        next_state, reward, terminal = self.cpt_env.step(action_index)
        self.step_count += 1

        if terminal or self.step_count >= self.max_steps:
            self.game_over = True

        return next_state, reward, terminal

    def render_plots_to_surface(self):
        """Render matplotlib plots to pygame surface"""
        # Create figure with fixed size and consistent spacing
        fig = plt.figure(figsize=(10, 8))

        # Create a layout with space reserved for the colorbar - adjust ratio for better boundaries
        gs = fig.add_gridspec(3, 2, width_ratios=[20, 1], height_ratios=[1, 1, 1],
                             hspace=0.3, wspace=0.02)

        # Create axes for plots and a separate axis for colorbar
        axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        cbar_ax = fig.add_subplot(gs[:, 1])

        # Get data
        all_x = np.unique(self.current_data[:, 0])
        all_y = np.unique(self.current_data[:, 1])

        # Create consistent field dimensions for visualization
        field_width = self.cpt_env.image_width
        field_height = len(all_y)

        # Create empty data arrays with proper dimensions if needed
        if self.cpt_env.predicted_data is None:
            empty_predicted = np.zeros((field_width, field_height))
        else:
            empty_predicted = self.cpt_env.predicted_data

        if self.cpt_env.true_data is None:
            empty_true = np.zeros((field_width, field_height))
        else:
            empty_true = self.cpt_env.true_data

        # Plot 1: Sampled positions
        axs[0].set_title("Sampled CPT Locations")
        for pos, dat in zip(self.cpt_env.sampled_positions, self.cpt_env.sampled_values):
            sc = axs[0].scatter(np.ones(len(all_y)) * pos, all_y, c=dat,
                          vmin=0, vmax=4.5, marker="s", s=3, cmap="viridis")
        axs[0].set_ylabel("Depth")
        axs[0].set_xlim([0, field_width])
        axs[0].set_ylim([0, np.max(all_y)])
        axs[0].grid(True)

        # Plot 2: Interpolated field (always shown with consistent dimensions)
        axs[1].set_title("Interpolated Field")
        im = axs[1].imshow(empty_predicted.T, vmin=0, vmax=4.5, cmap="viridis",
                    extent=[0, np.max(all_x), np.max(all_y), 0])
        axs[1].set_ylabel("Depth")
        axs[1].set_xlim([0, field_width])
        axs[1].grid(True)
        axs[1].invert_yaxis()

        # Plot 3: True field (always shown with consistent dimensions)
        axs[2].set_title("True Field")
        im = axs[2].imshow(empty_true.T, vmin=0, vmax=4.5, cmap="viridis",
                    extent=[0, np.max(all_x), np.max(all_y), 0])
        axs[2].set_xlabel("Distance [m]")
        axs[2].set_ylabel("Depth")
        axs[2].set_xlim([0, field_width])
        axs[2].grid(True)
        axs[2].invert_yaxis()

        # Always add colorbar to the reserved axis
        fig.colorbar(im, cax=cbar_ax, label="IC")

        # Add an attribute to track the width ratio for later calculations
        self.plot_width_ratio = gs.get_width_ratios()[0] / (gs.get_width_ratios()[0] + gs.get_width_ratios()[1])

        # Convert plot to pygame surface with fixed size
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plot_surface = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
        plt.close(fig)

        # Store the actual width of the plot content and data boundaries
        self.plot_actual_width = field_width
        self.data_min_x = 0  # The minimum x-value in data coordinates
        self.data_max_x = field_width  # The maximum x-value in data coordinates

        # Create a consistent function for mapping between coordinate systems
        # This ensures the same mapping is used everywhere
        self.data_to_screen = lambda data_x: int(self.plot_start_x +
            ((data_x - self.data_min_x) / (self.data_max_x - self.data_min_x)) * self.plot_usable_width)

        self.screen_to_data = lambda screen_x: int(self.data_min_x +
            ((screen_x - self.plot_start_x) / self.plot_usable_width) * (self.data_max_x - self.data_min_x))

        return plot_surface

    def render_start_screen(self):
        """Render the start screen with credits and start button"""
        # Fill the background
        self.screen.fill(self.WHITE)

        # Title
        title_text = self.title_font.render("CPT Super Learn", True, self.DARK_BLUE)
        title_rect = title_text.get_rect(center=(self.screen_width//2, 120))
        self.screen.blit(title_text, title_rect)

        subtitle_text = self.font.render("Digital Twin for CPT Sampling Strategy", True, self.BLACK)
        subtitle_rect = subtitle_text.get_rect(center=(self.screen_width//2, 180))
        self.screen.blit(subtitle_text, subtitle_rect)

        # Credits
        credits = [
            "© 2023 CPT Super Learn Team",
            "Developed as an interactive digital twin for CPT sampling",
            "Using reinforcement learning to optimize CPT placement",
            "",
            "Controls:",
            "M: Toggle Manual/AI mode",
            "N: New Game",
            "Arrow keys: Move cursor (Manual mode)",
            "Space: Place CPT",
            "ESC: Quit"
        ]

        y_pos = 250
        for line in credits:
            credit_text = self.small_font.render(line, True, self.BLACK)
            credit_rect = credit_text.get_rect(center=(self.screen_width//2, y_pos))
            self.screen.blit(credit_text, credit_rect)
            y_pos += 30

        # Start button - check if mouse is hovering
        mouse_pos = pygame.mouse.get_pos()
        self.start_button['hovered'] = self.start_button['rect'].collidepoint(mouse_pos)
        button_color = self.start_button['hover_color'] if self.start_button['hovered'] else self.start_button['color']

        # Draw button
        pygame.draw.rect(self.screen, button_color, self.start_button['rect'], border_radius=10)
        pygame.draw.rect(self.screen, self.BLACK, self.start_button['rect'], 2, border_radius=10)

        # Button text
        button_text = self.font.render(self.start_button['text'], True, self.WHITE)
        button_text_rect = button_text.get_rect(center=self.start_button['rect'].center)
        self.screen.blit(button_text, button_text_rect)

        pygame.display.flip()

    def render(self):
        """Render the game"""
        # If in start screen, render that instead
        if self.in_start_screen:
            self.render_start_screen()
            return

        self.screen.fill(self.WHITE)

        # Draw plots if data is loaded
        if self.current_data is not None:
            plot_surface = self.render_plots_to_surface()
            # Fixed size for plot area to maintain consistent dimensions
            plot_area = (int(self.field_width), int(self.screen_height * 0.7))
            self.screen.blit(pygame.transform.scale(plot_surface, plot_area), (self.plot_left_margin, 80))

            # Calculate the exact plot area (figure boundaries)
            # More accurate calculation of where the actual plot area is within the figure
            main_plot_width = self.plot_width * self.plot_width_ratio

            # Adjust margins for better accuracy - use percentages that match matplotlib's default layout
            plot_content_margin_left = self.plot_width * 0.13  # Left margin including y-axis labels and ticks
            plot_content_margin_right = self.plot_width * 0.03  # Right margin before colorbar

            self.plot_start_x = self.plot_left_margin + plot_content_margin_left
            self.plot_end_x = self.plot_left_margin + main_plot_width - plot_content_margin_right
            self.plot_usable_width = self.plot_end_x - self.plot_start_x

            # Define the coordinate mapping functions
            self.data_to_screen = lambda data_x: int(self.plot_start_x +
                ((data_x - self.data_min_x) / (self.data_max_x - self.data_min_x)) * self.plot_usable_width)

            self.screen_to_data = lambda screen_x: int(self.data_min_x +
                ((screen_x - self.plot_start_x) / self.plot_usable_width) * (self.data_max_x - self.data_min_x))

            # Calculate top figure's dimensions
            plot_start_y = 80
            total_plot_height = int(self.screen_height * 0.7)
            single_plot_height = total_plot_height / 3
            top_figure_top = plot_start_y
            top_figure_bottom = plot_start_y + single_plot_height
            title_margin = 25
            top_figure_axes_top = top_figure_top + title_margin

        # Draw game information
        title_text = self.font.render("CPT Super Learn - Digital Twin", True, self.BLACK)
        self.screen.blit(title_text, (20, 20))

        # File info
        if self.current_file:
            file_text = self.small_font.render(f"File: {self.current_file}", True, self.BLACK)
            self.screen.blit(file_text, (20, 50))

        # Mode info
        mode_text = self.small_font.render(f"Mode: {'AI' if self.auto_mode else 'Manual'}", True, self.BLACK)
        self.screen.blit(mode_text, (300, 50))

        # CPT count
        count_text = self.small_font.render(f"CPTs: {self.step_count}/{self.max_steps}", True, self.BLACK)
        self.screen.blit(count_text, (500, 50))

        # RMSE if available
        if hasattr(self.cpt_env, 'true_data') and hasattr(self.cpt_env, 'predicted_data') and self.cpt_env.true_data is not None:
            rmse = np.sqrt(np.mean((self.cpt_env.true_data - self.cpt_env.predicted_data) ** 2))
            rmse_text = self.small_font.render(f"RMSE: {rmse:.4f}", True, self.BLACK)
            self.screen.blit(rmse_text, (650, 50))

        # Draw control panel
        panel_y = self.screen_height - 100
        pygame.draw.rect(self.screen, (220, 220, 220), (20, panel_y, self.field_width, 80))

        # Instructions
        if not self.game_over:
            if not self.auto_mode:
                instructions = "Use LEFT/RIGHT arrows to move cursor, SPACE to place CPT"
            else:
                instructions = "Press SPACE to let AI place the next CPT"

            inst_text = self.small_font.render(instructions, True, self.BLACK)
            self.screen.blit(inst_text, (30, panel_y + 10))
        else:
            game_over_text = self.font.render("Game Over! Press 'N' for a new game", True, self.RED)
            self.screen.blit(game_over_text, (30, panel_y + 10))

        # Controls
        controls_text = self.small_font.render("M: Toggle Manual/AI mode | N: New Game | ESC: Quit", True, self.BLACK)
        self.screen.blit(controls_text, (30, panel_y + 40))

        # Draw truck in manual mode or auto mode
        if not self.game_over and self.current_data is not None:
            # Get current position in data space
            position = self.cursor_position if not self.auto_mode else self.cpt_env.sampled_positions[-1]

            # Use our consistent mapping function to convert data to screen coordinates
            cursor_x = self.data_to_screen(position)

            # Constrain to plot boundaries (without changing the cursor_position)
            cursor_x = max(self.plot_start_x, min(cursor_x, self.plot_end_x))

            # IMPORTANT: Do NOT update the cursor_position here to prevent drift
            # We only want to update cursor_position when user explicitly moves it with arrow keys

            # Draw truck at the top of the first plot (centered on position)
            truck_x = cursor_x - self.truck_img.get_width() // 2
            truck_y = top_figure_axes_top - self.truck_img.get_height() - 2
            self.screen.blit(self.truck_img, (truck_x, truck_y))

            # Draw dashed line from truck to indicate drilling point with improved visibility
            # Draw a more visible vertical line through the entire top plot
            y_start = top_figure_axes_top
            y_end = int(top_figure_bottom)

            # First draw a thin solid red line as background
            pygame.draw.line(self.screen, self.RED, (cursor_x, y_start), (cursor_x, y_end), 1)

            # Then overlay a dashed white pattern to create dashed effect
            dash_length = 4
            gap_length = 4
            y = y_start
            while y < y_end:
                end_y = min(y + gap_length, y_end)
                pygame.draw.line(self.screen, self.WHITE, (cursor_x, y), (cursor_x, end_y), 1)
                y += dash_length + gap_length

        pygame.display.flip()

    def run(self):
        """Main game loop"""
        # Don't load a file yet - wait until start screen is dismissed
        state = None

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    # Don't process other keys if we're in the start screen
                    if self.in_start_screen:
                        continue

                    if event.key == pygame.K_m:
                        # Toggle mode
                        self.auto_mode = not self.auto_mode
                    elif event.key == pygame.K_n:
                        # New game
                        state = self.load_random_file()
                    elif event.key == pygame.K_SPACE and not self.game_over:
                        if self.auto_mode:
                            # AI mode - let the agent decide
                            state, reward, terminal = self.auto_step(state)
                        else:
                            # Manual mode - place CPT exactly at cursor_position
                            state, reward, terminal = self.manual_step(self.cursor_position)

                    # Manual navigation keys - movement constrained to data boundaries
                    elif event.key == pygame.K_RIGHT and not self.auto_mode and not self.game_over:
                        step_size = max(1, self.cpt_env.image_width // 50)  # Adaptive step size
                        # Keep the cursor within the data boundaries
                        self.cursor_position = min(self.cursor_position + step_size, self.cpt_env.image_width - 1)
                    elif event.key == pygame.K_LEFT and not self.auto_mode and not self.game_over:
                        step_size = max(1, self.cpt_env.image_width // 50)  # Adaptive step size
                        # Keep the cursor within the data boundaries
                        self.cursor_position = max(self.cursor_position - step_size, 0)

                # Handle mouse clicks
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        # Check if start button was clicked
                        if self.in_start_screen and self.start_button['hovered']:
                            # Exit start screen and start the game
                            self.in_start_screen = False
                            state = self.load_random_file()

            # In start screen, just render that
            if self.in_start_screen:
                self.render_start_screen()
            else:
                # Normal game rendering
                self.render()

            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    validation_data_folder = "data/vali"
    model_path = "results_2"  # Path to the trained model

    game = CPTGame(validation_data_folder, model_path)
    game.run()