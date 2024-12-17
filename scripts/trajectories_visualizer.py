import gradio as gr
from PIL import Image
import torch
import random
from fire import Fire

def main(save_path: str):
    trajs = torch.load(f"{save_path}/evaluate_trajectories.pt")
    trajs = list(filter(lambda x: len(x) > 0, trajs))

    def update_label(i):
        task = trajs[i][0]["observation"]['task']

        images = []
        for step_id in range(len(trajs[i])):
            img = Image.open(trajs[i][step_id]['observation']['image'])
            img = img.convert("RGB")
            images.append(img)
        # If there are images, concatenate them into a single image for display
        if images:
            max_images_per_row=1
            # Calculate the number of rows needed
            num_rows = (len(images) + max_images_per_row - 1) // max_images_per_row
            
            # Prepare to create the final combined image
            row_images = []
            max_row_width = 0
            total_height = 0
            
            for row in range(num_rows):
                # Calculate the start and end index of images for this row
                start_index = row * max_images_per_row
                end_index = min(start_index + max_images_per_row, len(images))
                
                # Get the row images
                row_images_subset = images[start_index:end_index]
                
                # Calculate total width and maximum height for this row
                total_width = sum(img.width for img in row_images_subset)
                max_height = max(img.height for img in row_images_subset)
                
                # Create an image for this row
                row_image = Image.new('RGB', (total_width, max_height))
                x_offset = 0
                for img in row_images_subset:
                    row_image.paste(img, (x_offset, 0))
                    x_offset += img.width
                
                row_images.append(row_image)
                max_row_width = max(max_row_width, total_width)
                total_height += max_height
        
            # Create the final combined image
            combined_image = Image.new('RGB', (max_row_width, total_height))
            y_offset = 0
            for row_img in row_images:
                combined_image.paste(row_img, (0, y_offset))
                y_offset += row_img.height
        answer = trajs[i][-1]["action"]
        actions = ""
        for j, t in enumerate(trajs[i]):
            actions += f"{j}: {t['action']}\n"
        reference = trajs[i][-1]["reference"] if 'reference' in trajs[i][-1] else "No reference"
        evaluation = trajs[i][-1]['eval_info'] if 'eval_info' in trajs[i][-1] else "No evaluation info"
        # for j, t in enumerate(trajs[i]):
        #     print(t['eval_info'] if 'eval_info' in t else "No evaluation info")
        return task, actions, answer, reference, evaluation, str(trajs[i][-1]["reward"]),combined_image

    # Create the Gradio interface
    interface = gr.Interface(
        fn=update_label,
        inputs=[gr.components.Number(label="Index")],
        outputs=[
            gr.components.Text(label="Task"),
            gr.components.Text(label="Actions"),
            gr.components.Text(label="Answer"),
            gr.components.Text(label="Reference"),
            gr.components.Text(label="Evaluation"),
            gr.components.Text(label="Reward"),
            gr.Image(label="Images")
            ],
        title="Trajectory Visualizer",
        description="Change the index to see different trajectories."
    )

    # Launch the interface
    interface.launch(share=True, server_port=9781)

if __name__ == "__main__":
    Fire(main)