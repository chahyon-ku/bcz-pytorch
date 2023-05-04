from rlbench.const import colors
from rlbench.tasks.bcz_pick_and_place import SOURCE_NAMES, TARGET_NAMES

def get_task_embeddings(language_encoder):
    task_embeds = {'reach_target': [], 'pick_and_lift': [], 'pick_lemon': [], 'bcz_pick_and_place': []}
    for pair in colors:
        color_name, _ = pair
        language = 'reach the %s target' % color_name
        task_embed = language_encoder.encode([language])[0]
        task_embeds['reach_target'].append(task_embed)

    for pair in colors:
        color_name, _ = pair
        language = 'pick up the %s block and lift it up to the target' % color_name
        task_embed = language_encoder.encode([language])[0]
        task_embeds['pick_and_lift'].append(task_embed)
    
    task_embeds['pick_lemon'].append(language_encoder.encode(['pick the lemon'])[0])

    for i in range(5):
        for target_name in TARGET_NAMES:
            for source_name in SOURCE_NAMES:
                language = f'place the {source_name} in the {target_name}'
                task_embed = language_encoder.encode([language])[0]
                task_embeds['bcz_pick_and_place'].append(task_embed)
        
    task_embeds['bcz_pick_and_place_tight'] = task_embeds['bcz_pick_and_place']
    task_embeds['bcz'] = task_embeds['bcz_pick_and_place']

    return task_embeds