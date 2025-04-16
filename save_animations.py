import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

def animate_path_on_image(image_path, coordinates, obstaclesList, path, output_gif="animation.gif", interval=500, dpi=150):
    """
    Create an animation of a constructed path drawn over a given image background,
    and save the animation as a high-quality gif.

    Parameters:
      image_path: str
          Path to the background image file.
      coordinates: list of [x, y]
          List of vertex coordinates (index 0 is unused; vertices are indexed from 1..N).
      path: list of int
          List of vertex indices representing the constructed path from start to goal.
      output_gif: str, optional
          Filename for the output gif (default "animation.gif").
      interval: int, optional
          Time (in milliseconds) between frames of the animation (default 500).
      dpi: int, optional
          Dots per inch for the saved gif (default 150).
    """
    img_bgr = cv2.imread(image_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    obs = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    for obstacle in obstaclesList:
        obs = cv2.circle(obs, (int(obstacle[0]), int(obstacle[1])), obstacle[2], (250, 0, 0), -1)

    img = cv2.addWeighted(obs, 0.7, img, 0.3, 0)
    N= len(coordinates) - 1
    cv2.drawMarker(img, (coordinates[N][0], coordinates[N][1]), (0, 0, 0), 1, markerSize=12, thickness=3)
    cv2.circle(img, (coordinates[N - 1][0], coordinates[N - 1][1]), 3, (0, 0, 0), thickness=3)

    # Create a figure and axis.
    fig, ax = plt.subplots()

    # Display the image as the background.
    # Set extent so that the image covers [0, width] in x and [0, height] in y.
    ax.imshow(img)

    # Extract the path coordinates.
    path_coords = [coordinates[v] for v in path]
    xs, ys = zip(*path_coords)

    # Create an empty line and marker for the path.
    line, = ax.plot([], [], 'ro-', lw=2, ms=2)

    # Set the axis limits based on the image size.
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title("Path Search Animation")

    # Initialization function for the animation.
    def init():
        line.set_data([], [])
        return line,

    # Update function to draw more of the path at each frame.
    def update(frame):
        current_x = xs[:frame + 1]
        current_y = ys[:frame + 1]
        line.set_data(current_x, current_y)
        return line,

    # Create the animation.
    ani = animation.FuncAnimation(fig, update, frames=len(xs), init_func=init,
                                  interval=interval, blit=True, repeat=False)

    # Save the animation as a gif using PillowWriter.
    writer = animation.PillowWriter(fps=1000 // interval, metadata=dict(artist='PathPlanner'), bitrate=-1)
    ani.save(output_gif, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Animation saved as {output_gif}")

def animate_bidirectional_path_on_image(image_path, coordinates, obstaclesList, path_forward, path_backward,
                                        output_gif="bidirectional_animation.gif", interval=500, dpi=150):
    """
    Animate a bidirectional search (e.g., from Bidirectional A* or RRT-Connect)
    that constructs two branches: one from the start and another from the goal,
    growing toward the meeting point.

    Parameters:
      image_path: str
          Path to the background image file.
      coordinates: list of [x, y]
          List of vertex coordinates (index 0 is unused; vertices are indexed from 1..N).
      path_forward: list of int
          List of vertex indices from start to the meeting point.
      path_backward: list of int
          List of vertex indices from the meeting point to the goal.
          (It will be reversed so that the growth appears from the goal toward the meeting point.)
      output_gif: str, optional
          Filename for the output gif (default "bidirectional_animation.gif").
      interval: int, optional
          Time (in milliseconds) between frames of the animation (default 500).
      dpi: int, optional
          Dots per inch for the saved gif (default 150).
    """
    # Load the background image.
    img_bgr = cv2.imread(image_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    obs = img.copy()
    for obstacle in obstaclesList:
        obs = cv2.circle(obs, (int(obstacle[0]), int(obstacle[1])), obstacle[2], (250, 0, 0), -1)

    img = cv2.addWeighted(obs, 0.7, img, 0.3, 0)

    cv2.drawMarker(img, (coordinates[N][0], coordinates[N][1]), (0, 0, 0), 1, markerSize=12, thickness=3)
    cv2.circle(img, (coordinates[N - 1][0], coordinates[N - 1][1]), 3, (0, 0, 0), thickness=3)

    # Create a figure and axis.
    fig, ax = plt.subplots()
    # Display the image as background (assume the image coordinate system is such that
    # x ranges from 0 to width and y ranges from 0 to height).
    ax.imshow(img)

    # Prepare the coordinates for the two branches.
    # For the forward branch, extract coordinates from start to meeting point.
    forward_coords = [coordinates[v] for v in path_forward]
    f_xs, f_ys = zip(*forward_coords)

    # For the backward branch, reverse it so that it will be drawn from goal to meeting point.
    backward_branch = list(reversed(path_backward))
    b_coords = [coordinates[v] for v in backward_branch]
    b_xs, b_ys = zip(*b_coords)

    # Create line objects for both branches.
    # Here, we use one marker+line style for the forward branch and a different one for the backward branch.
    forward_line, = ax.plot([], [], 'ro-', lw=2, ms=2, label="Forward (start -> meeting)")
    backward_line, = ax.plot([], [], 'ko-', lw=2, ms=2, label="Backward (goal -> meeting)")

    # Optionally, set axis limits based on image dimensions.
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title("Bidirectional Path Search Animation")
    ax.legend(loc="upper right")

    # Determine the total number of frames: use the larger branch length.
    total_frames = max(len(f_xs), len(b_xs))

    # Initialization function: clear both lines.
    def init():
        forward_line.set_data([], [])
        backward_line.set_data([], [])
        return forward_line, backward_line

    # Update function: For each frame, show more of each branch.
    def update(frame):
        # For forward branch, if we haven't reached the end yet, show frame+1 points.
        if frame < len(f_xs):
            current_forward_x = f_xs[:frame + 1]
            current_forward_y = f_ys[:frame + 1]
        else:
            # Otherwise, show the entire forward branch.
            current_forward_x = f_xs
            current_forward_y = f_ys
        forward_line.set_data(current_forward_x, current_forward_y)

        # For backward branch, similarly animate the branch growing from goal.
        if frame < len(b_xs):
            current_backward_x = b_xs[:frame + 1]
            current_backward_y = b_ys[:frame + 1]
        else:
            current_backward_x = b_xs
            current_backward_y = b_ys
        backward_line.set_data(current_backward_x, current_backward_y)

        return forward_line, backward_line

    # Create the animation.
    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init,
                                  interval=interval, blit=True, repeat=False)

    # Save the animation as a gif with high quality using PillowWriter.
    writer = animation.PillowWriter(fps=1000 // interval, metadata=dict(artist='PathPlanner'), bitrate=-1)
    ani.save(output_gif, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Bidirectional path animation saved as {output_gif}")
