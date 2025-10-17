class SettingsObj():
    def __init__(self, **setting):
        for key, value in setting.items():
            setattr(self, key, value)


meta_data_hand = '''Pelvis (x)', 'Pelvis (y)', 'Pelvis (z)', 'L5 (x)', 'L5 (y)', 'L5 (z)', 'L3 (x)', 'L3 (y)', 'L3 (z)', 'T12 (x)', 'T12 (y)', 'T12 (z)', 'T8 (x)', 'T8 (y)', 'T8 (z)', 'Neck (x)', 'Neck (y)', 'Neck (z)', 'Head (x)', 'Head (y)', 'Head (z)', 'Right Shoulder (x)', 'Right Shoulder (y)', 'Right Shoulder (z)', 'Right Upper Arm (x)', 'Right Upper Arm (y)', 'Right Upper Arm (z)', 'Right Forearm (x)', 'Right Forearm (y)', 'Right Forearm (z)', 'Right Hand (x)', 'Right Hand (y)', 'Right Hand (z)', 'Left Shoulder (x)', 'Left Shoulder (y)', 'Left Shoulder (z)', 'Left Upper Arm (x)', 'Left Upper Arm (y)', 'Left Upper Arm (z)', 'Left Forearm (x)', 'Left Forearm (y)', 'Left Forearm (z)', 'Left Hand (x)', 'Left Hand (y)', 'Left Hand (z)', 'Right Upper Leg (x)', 'Right Upper Leg (y)', 'Right Upper Leg (z)', 'Right Lower Leg (x)', 'Right Lower Leg (y)', 'Right Lower Leg (z)', 'Right Foot (x)', 'Right Foot (y)', 'Right Foot (z)', 'Right Toe (x)', 'Right Toe (y)', 'Right Toe (z)', 'Left Upper Leg (x)', 'Left Upper Leg (y)', 'Left Upper Leg (z)', 'Left Lower Leg (x)', 'Left Lower Leg (y)', 'Left Lower Leg (z)', 'Left Foot (x)', 'Left Foot (y)', 'Left Foot (z)', 'Left Toe (x)', 'Left Toe (y)', 'Left Toe (z)', 'Left Carpus (x)', 'Left Carpus (y)', 'Left Carpus (z)', 'Left First Metacarpal (x)', 'Left First Metacarpal (y)', 'Left First Metacarpal (z)', 'Left First Proximal Phalange (x)', 'Left First Proximal Phalange (y)', 'Left First Proximal Phalange (z)', 'Left First Distal Phalange (x)', 'Left First Distal Phalange (y)', 'Left First Distal Phalange (z)', 'Left Second Metacarpal (x)', 'Left Second Metacarpal (y)', 'Left Second Metacarpal (z)', 'Left Second Proximal Phalange (x)', 'Left Second Proximal Phalange (y)', 'Left Second Proximal Phalange (z)', 'Left Second Middle Phalange (x)', 'Left Second Middle Phalange (y)', 'Left Second Middle Phalange (z)', 'Left Second Distal Phalange (x)', 'Left Second Distal Phalange (y)', 'Left Second Distal Phalange (z)', 'Left Third Metacarpal (x)', 'Left Third Metacarpal (y)', 'Left Third Metacarpal (z)', 'Left Third Proximal Phalange (x)', 'Left Third Proximal Phalange (y)', 'Left Third Proximal Phalange (z)', 'Left Third Middle Phalange (x)', 'Left Third Middle Phalange (y)', 'Left Third Middle Phalange (z)', 'Left Third Distal Phalange (x)', 'Left Third Distal Phalange (y)', 'Left Third Distal Phalange (z)', 'Left Fourth Metacarpal (x)', 'Left Fourth Metacarpal (y)', 'Left Fourth Metacarpal (z)', 'Left Fourth Proximal Phalange (x)', 'Left Fourth Proximal Phalange (y)', 'Left Fourth Proximal Phalange (z)', 'Left Fourth Middle Phalange (x)', 'Left Fourth Middle Phalange (y)', 'Left Fourth Middle Phalange (z)', 'Left Fourth Distal Phalange (x)', 'Left Fourth Distal Phalange (y)', 'Left Fourth Distal Phalange (z)', 'Left Fifth Metacarpal (x)', 'Left Fifth Metacarpal (y)', 'Left Fifth Metacarpal (z)', 'Left Fifth Proximal Phalange (x)', 'Left Fifth Proximal Phalange (y)', 'Left Fifth Proximal Phalange (z)', 'Left Fifth Middle Phalange (x)', 'Left Fifth Middle Phalange (y)', 'Left Fifth Middle Phalange (z)', 'Left Fifth Distal Phalange (x)', 'Left Fifth Distal Phalange (y)', 'Left Fifth Distal Phalange (z)', 'Right Carpus (x)', 'Right Carpus (y)', 'Right Carpus (z)', 'Right First Metacarpal (x)', 'Right First Metacarpal (y)', 'Right First Metacarpal (z)', 'Right First Proximal Phalange (x)', 'Right First Proximal Phalange (y)', 'Right First Proximal Phalange (z)', 'Right First Distal Phalange (x)', 'Right First Distal Phalange (y)', 'Right First Distal Phalange (z)', 'Right Second Metacarpal (x)', 'Right Second Metacarpal (y)', 'Right Second Metacarpal (z)', 'Right Second Proximal Phalange (x)', 'Right Second Proximal Phalange (y)', 'Right Second Proximal Phalange (z)', 'Right Second Middle Phalange (x)', 'Right Second Middle Phalange (y)', 'Right Second Middle Phalange (z)', 'Right Second Distal Phalange (x)', 'Right Second Distal Phalange (y)', 'Right Second Distal Phalange (z)', 'Right Third Metacarpal (x)', 'Right Third Metacarpal (y)', 'Right Third Metacarpal (z)', 'Right Third Proximal Phalange (x)', 'Right Third Proximal Phalange (y)', 'Right Third Proximal Phalange (z)', 'Right Third Middle Phalange (x)', 'Right Third Middle Phalange (y)', 'Right Third Middle Phalange (z)', 'Right Third Distal Phalange (x)', 'Right Third Distal Phalange (y)', 'Right Third Distal Phalange (z)', 'Right Fourth Metacarpal (x)', 'Right Fourth Metacarpal (y)', 'Right Fourth Metacarpal (z)', 'Right Fourth Proximal Phalange (x)', 'Right Fourth Proximal Phalange (y)', 'Right Fourth Proximal Phalange (z)', 'Right Fourth Middle Phalange (x)', 'Right Fourth Middle Phalange (y)', 'Right Fourth Middle Phalange (z)', 'Right Fourth Distal Phalange (x)', 'Right Fourth Distal Phalange (y)', 'Right Fourth Distal Phalange (z)', 'Right Fifth Metacarpal (x)', 'Right Fifth Metacarpal (y)', 'Right Fifth Metacarpal (z)', 'Right Fifth Proximal Phalange (x)', 'Right Fifth Proximal Phalange (y)', 'Right Fifth Proximal Phalange (z)', 'Right Fifth Middle Phalange (x)', 'Right Fifth Middle Phalange (y)', 'Right Fifth Middle Phalange (z)', 'Right Fifth Distal Phalange (x)', 'Right Fifth Distal Phalange (y)', 'Right Fifth Distal Phalange (z)'''
# meta_data_hand = '''Left Carpus (x)', 'Left Carpus (y)', 'Left Carpus (z)', 'Left First Metacarpal (x)', 'Left First Metacarpal (y)', 'Left First Metacarpal (z)', 'Left First Proximal Phalange (x)', 'Left First Proximal Phalange (y)', 'Left First Proximal Phalange (z)', 'Left First Distal Phalange (x)', 'Left First Distal Phalange (y)', 'Left First Distal Phalange (z)', 'Left Second Metacarpal (x)', 'Left Second Metacarpal (y)', 'Left Second Metacarpal (z)', 'Left Second Proximal Phalange (x)', 'Left Second Proximal Phalange (y)', 'Left Second Proximal Phalange (z)', 'Left Second Middle Phalange (x)', 'Left Second Middle Phalange (y)', 'Left Second Middle Phalange (z)', 'Left Second Distal Phalange (x)', 'Left Second Distal Phalange (y)', 'Left Second Distal Phalange (z)', 'Left Third Metacarpal (x)', 'Left Third Metacarpal (y)', 'Left Third Metacarpal (z)', 'Left Third Proximal Phalange (x)', 'Left Third Proximal Phalange (y)', 'Left Third Proximal Phalange (z)', 'Left Third Middle Phalange (x)', 'Left Third Middle Phalange (y)', 'Left Third Middle Phalange (z)', 'Left Third Distal Phalange (x)', 'Left Third Distal Phalange (y)', 'Left Third Distal Phalange (z)', 'Left Fourth Metacarpal (x)', 'Left Fourth Metacarpal (y)', 'Left Fourth Metacarpal (z)', 'Left Fourth Proximal Phalange (x)', 'Left Fourth Proximal Phalange (y)', 'Left Fourth Proximal Phalange (z)', 'Left Fourth Middle Phalange (x)', 'Left Fourth Middle Phalange (y)', 'Left Fourth Middle Phalange (z)', 'Left Fourth Distal Phalange (x)', 'Left Fourth Distal Phalange (y)', 'Left Fourth Distal Phalange (z)', 'Left Fifth Metacarpal (x)', 'Left Fifth Metacarpal (y)', 'Left Fifth Metacarpal (z)', 'Left Fifth Proximal Phalange (x)', 'Left Fifth Proximal Phalange (y)', 'Left Fifth Proximal Phalange (z)', 'Left Fifth Middle Phalange (x)', 'Left Fifth Middle Phalange (y)', 'Left Fifth Middle Phalange (z)', 'Left Fifth Distal Phalange (x)', 'Left Fifth Distal Phalange (y)', 'Left Fifth Distal Phalange (z)', 'Right Carpus (x)', 'Right Carpus (y)', 'Right Carpus (z)', 'Right First Metacarpal (x)', 'Right First Metacarpal (y)', 'Right First Metacarpal (z)', 'Right First Proximal Phalange (x)', 'Right First Proximal Phalange (y)', 'Right First Proximal Phalange (z)', 'Right First Distal Phalange (x)', 'Right First Distal Phalange (y)', 'Right First Distal Phalange (z)', 'Right Second Metacarpal (x)', 'Right Second Metacarpal (y)', 'Right Second Metacarpal (z)', 'Right Second Proximal Phalange (x)', 'Right Second Proximal Phalange (y)', 'Right Second Proximal Phalange (z)', 'Right Second Middle Phalange (x)', 'Right Second Middle Phalange (y)', 'Right Second Middle Phalange (z)', 'Right Second Distal Phalange (x)', 'Right Second Distal Phalange (y)', 'Right Second Distal Phalange (z)', 'Right Third Metacarpal (x)', 'Right Third Metacarpal (y)', 'Right Third Metacarpal (z)', 'Right Third Proximal Phalange (x)', 'Right Third Proximal Phalange (y)', 'Right Third Proximal Phalange (z)', 'Right Third Middle Phalange (x)', 'Right Third Middle Phalange (y)', 'Right Third Middle Phalange (z)', 'Right Third Distal Phalange (x)', 'Right Third Distal Phalange (y)', 'Right Third Distal Phalange (z)', 'Right Fourth Metacarpal (x)', 'Right Fourth Metacarpal (y)', 'Right Fourth Metacarpal (z)', 'Right Fourth Proximal Phalange (x)', 'Right Fourth Proximal Phalange (y)', 'Right Fourth Proximal Phalange (z)', 'Right Fourth Middle Phalange (x)', 'Right Fourth Middle Phalange (y)', 'Right Fourth Middle Phalange (z)', 'Right Fourth Distal Phalange (x)', 'Right Fourth Distal Phalange (y)', 'Right Fourth Distal Phalange (z)', 'Right Fifth Metacarpal (x)', 'Right Fifth Metacarpal (y)', 'Right Fifth Metacarpal (z)', 'Right Fifth Proximal Phalange (x)', 'Right Fifth Proximal Phalange (y)', 'Right Fifth Proximal Phalange (z)', 'Right Fifth Middle Phalange (x)', 'Right Fifth Middle Phalange (y)', 'Right Fifth Middle Phalange (z)', 'Right Fifth Distal Phalange (x)', 'Right Fifth Distal Phalange (y)', 'Right Fifth Distal Phalange (z)'''
meta_data_hand = meta_data_hand.replace("\'", "")
meta_data_hand = meta_data_hand.replace(' (x)', "")
meta_data_hand = meta_data_hand.split(', ')
meta_data_hand = [x for x in meta_data_hand if not x.endswith(')')]
hands_data_start_index = meta_data_hand.index('Left Carpus')

HAND_CHAIN_NAMES = {
    # 'palm': ['Carpus', 'First Metacarpal', 'Second Metacarpal', 'Third Metacarpal', 'Fourth Metacarpal', 'Fifth Metacarpal'],
    'thumb': ['Carpus', 'First Metacarpal','First Proximal Phalange', 'First Distal Phalange'],
    'index': ['Carpus', 'Second Metacarpal','Second Proximal Phalange', 'Second Middle Phalange', 'Second Distal Phalange'],
    'middle': ['Carpus', 'Third Metacarpal', 'Third Proximal Phalange', 'Third Middle Phalange', 'Third Distal Phalange'],
    'ring': ['Carpus', 'Fourth Metacarpal', 'Fourth Proximal Phalange', 'Fourth Middle Phalange', 'Fourth Distal Phalange'],
    'pinky': ['Carpus', 'Fifth Metacarpal', 'Fifth Proximal Phalange', 'Fifth Middle Phalange', 'Fifth Distal Phalange']
}

FINGER_SEGMENT_NAMES = [
    'Carpus',
    'First Metacarpal',
    'First Proximal Phalange',
    'First Distal Phalange',
    'Second Metacarpal',
    'Second Proximal Phalange',
    'Second Middle Phalange',
    'Second Distal Phalange',
    'Third Metacarpal',
    'Third Proximal Phalange',
    'Third Middle Phalange',
    'Third Distal Phalange',
    'Fourth Metacarpal',
    'Fourth Proximal Phalange',
    'Fourth Middle Phalange',
    'Fourth Distal Phalange',
    'Fifth Metacarpal',
    'Fifth Proximal Phalange',
    'Fifth Middle Phalange',
    'Fifth Distal Phalange',
    ]

LEFT_FINGER_SEGMENT_NAMES  = ['Left %s' % segment_label for segment_label in FINGER_SEGMENT_NAMES]

RIGHT_FINGER_SEGMENT_NAMES = ['Right %s' % segment_label for segment_label in FINGER_SEGMENT_NAMES]

LEFT_FINGER_TREESTR_LIST_INDICES = []

RIGHT_FINGER_TREESTR_LIST_INDICES = []

LEFT_HAND_CHAIN_NAMES = {}; RIGHT_HAND_CHAIN_NAMES = {}
for chain_name, segment_labels in HAND_CHAIN_NAMES.items():
    LEFT_HAND_CHAIN_NAMES[chain_name] = ['Left %s' % segment_label for segment_label in segment_labels]
    RIGHT_HAND_CHAIN_NAMES[chain_name] = ['Right %s' % segment_label for segment_label in segment_labels]

for chain_name, chain_labels in LEFT_HAND_CHAIN_NAMES.items():
    segment_indexes = []
    for chain_label in chain_labels:
        segment_indexes.append(meta_data_hand.index(chain_label))
    LEFT_FINGER_TREESTR_LIST_INDICES.extend(segment_indexes)

for chain_name, chain_labels in RIGHT_HAND_CHAIN_NAMES.items():
    segment_indexes = []
    for chain_label in chain_labels:
        segment_indexes.append(meta_data_hand.index(chain_label))
    RIGHT_FINGER_TREESTR_LIST_INDICES.extend(segment_indexes)

JOINT_CHAIN_NAMES = {
    'Left Leg':  ['Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe'],
    'Right Leg': ['Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe'],
    'Spine':     ['Head', 'Neck', 'T8', 'T12', 'L3', 'L5', 'Pelvis'], # top down
    'Hip':       ['Left Upper Leg', 'Pelvis', 'Right Upper Leg'],
    'Shoulders': ['Left Upper Arm', 'Left Shoulder', 'Right Shoulder', 'Right Upper Arm'],
    'Left Arm':  ['Left Upper Arm', 'Left Forearm', 'Left Hand'],
    'Right Arm': ['Right Upper Arm', 'Right Forearm', 'Right Hand'],
}

JOINT_SEGMENT_NAMES = [
    'Pelvis',
    'L5',
    'L3',
    'T12',
    'T8',
    'Neck',
    'Head',
    'Right Shoulder',
    'Right Upper Arm',
    'Right Forearm',
    'Right Hand',
    'Left Shoulder',
    'Left Upper Arm',
    'Left Forearm',
    'Left Hand',
    'Right Upper Leg',
    'Right Lower Leg',
    'Right Foot',
    'Right Toe',
    'Left Upper Leg',
    'Left Lower Leg',
    'Left Foot',
    'Left Toe',
]

JOINT_TREESTR_LIST_INDICES = []
for (chain_name, chain_labels) in JOINT_CHAIN_NAMES.items():
    segment_indexes = []
    for chain_label in chain_labels:
        segment_indexes.append(JOINT_SEGMENT_NAMES.index(chain_label))
    JOINT_TREESTR_LIST_INDICES.extend(segment_indexes)

LABEL_NAMES = [
    'Get/replace items from refrigerator/cabinets/drawers',
    'Peel a cucumber',
    'Clear cutting board',
    'Slice a cucumber',
    'Peel a potato',
    'Slice a potato',
    'Slice bread',
    'Spread almond butter on a bread slice',
    'Spread jelly on a bread slice',
    'Open/close a jar of almond butter',
    'Pour water from a pitcher into a glass',
    'Clean a plate with a sponge',
    'Clean a plate with a towel',
    'Clean a pan with a sponge',
    'Clean a pan with a towel',
    'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    'Stack on table: 3 each large/small plates, bowls',
    'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    'None',
]

LABELS = [
    'Fetch: Various',
    'Peel: Cucumber',
    'Clear Cutting Board',
    'Slice: Cucumber',
    'Peel: Potato',
    'Slice: Potato',
    'Slice: Bread',
    'Spread: Almond Butter',
    'Spread: Jelly',
    'Open or Close Jar',
    'Pour Water',
    'Clean: Plate, Sponge',
    'Clean: Plate, Towel',
    'Clean: Pan, Sponge',
    'Clean: Pan, Towel',
    'Fetch: Tableware',
    'Set Table',
    'Stack Tableware',
    'Dishwasher: Load',
    'Dishwasher: Unload',
    'None',
]

DATA_SETTINGS = {
    "src_dir" : "./raw_data/",
    "output_dir" : "./Dataset/",
    "normalize" : True,
    "denoise" : True,
    "myo_resample" : 10,
    "joint_resample" : 10,
    "tactile_resample" : 10,
    "gaze_resample": 10,
    "second_per_samples": 5,
    "sample_per_class" : 50,
    "tactile_aggregate" : False, 
    "hand_start_indices": meta_data_hand.index('Left Carpus'), 
}

DATA_SETTINGS = SettingsObj(**DATA_SETTINGS)

LABEL_ALTERNATES = {
    'None':[
        "No action",
        "Not defined",
        "Unknown",
        "Can not recognized."
    ],
    'Get/replace items from refrigerator/cabinets/drawers':[
        'Get/replace items from refrigerator/cabinets/drawers',
        "Retrieve or exchange items stored in the refrigerator, cabinets, or drawers.",
        "Obtain or substitute items from the refrigerator, cabinets, or drawers.",
        "Access or swap out items from the refrigerator, cabinets, or drawers.",
    ],
    'Peel a cucumber':[
        "Peel a cucumber.",
        "Remove the skin from a cucumber.",
        "Strip the outer layer of a cucumber.",
        "Skin a cucumber."

    ],
    'Clear cutting board':[
        "Clear the cutting board.",
        "Clean the cutting board.",
        "Wipe down the cutting board.",
        "Remove debris from the cutting board.",

    ],

    'Slice a cucumber':[
        "Slice a cucumber.",
        "Cut a cucumber into thin slices.",
        "Create cucumber slices.",
        "Prepare cucumber by cutting it into rounds."
    ],

    'Peel a potato':[
        "Peel a potato.",
        "Remove the skin from a potato.",
        "Strip the outer layer of a potato.",
        "Skin a potato."
    ],

    'Slice a potato':[
        "Slice a potato.",
        "Cut a potato into thin rounds.",
        "Prepare potato slices.",
        "Create potato rounds.",
        
    ],

    'Slice bread':[
        "Slice bread.",
        "Cut the bread into thin pieces.",
        "Prepare bread slices.",
        "Create thin bread sections."
    ],

    'Spread almond butter on a bread slice':[
        "Spread almond butter on a bread slice.",
        "Apply almond butter to a slice of bread.",
        "Coat a piece of bread with almond butter.",
        "Smear almond butter onto a bread slice."
    ],

    'Spread jelly on a bread slice':[
        "Apply jelly to a slice of bread.",
        "Distribute jelly evenly over a piece of bread.",
        "Smooth jelly onto a bread slice.",
        "Cover a slice of bread with a layer of jelly."
    ],

    'Open/close a jar of almond butter':[
        "Open or close a jar of almond butter.",
        "Unscrew or seal a container of almond butter.",
        "Loosen or secure the lid on a jar of almond butter.",
        "Remove or put on the cap of the almond butter jar."
    ],
    'Pour water from a pitcher into a glass':[
        "Pour water from a pitcher into a glass.",
        "Transfer water from a pitcher into a glass.",
        "Fill a glass with water from a pitcher.",
        "Empty the contents of a pitcher into a glass.",
    ],
    'Clean a plate with a sponge':[
        "Clean a plate with a sponge.",
        "Use a sponge to wash a plate.",
        "Scrub a plate with a sponge to remove dirt and grime.",
        "Employ a sponge to sanitize a plate."
    ],
    'Clean a plate with a towel':[
        "Clean a plate with a towel.",
        "Wipe a plate using a towel to make it spotless.",
        "Utilize a towel to remove any residue from a plate.",
        "Dry and polish a plate with a towel to ensure it's clean."
    ],
    'Clean a pan with a sponge':[
        "Clean a pan with a sponge.",
        "Use a sponge to scrub a pan until it's clean.",
        "Employ a sponge to remove residue and grease from a pan.",
        "Wash a pan using a sponge to ensure it's spotless."
    ],
    'Clean a pan with a towel':[
        'Clean a pan with a towel',
        "Wipe down a pan using a towel to remove any residue.",
        "Polish a pan with a towel to ensure it's free from any remaining debris.",
        "Use a towel to dry and cleanse the pan thoroughly."
    ],
    'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils':[
        "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils.",
        "Retrieve from the cabinets: 3 large and 3 small plates, 3 bowls, 3 mugs, 3 glasses, and sets of utensils.",
        "Access your cabinets to collect 3 large plates, 3 small plates, 3 bowls, 3 mugs, 3 glasses, and sets of utensils.",
        "Gather 3 large and 3 small plates, along with 3 bowls, 3 mugs, 3 glasses, and sets of utensils, from your cabinets."
    ],
    'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils':[
        "Set the table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils.",
        "Arrange the table with 3 large and 3 small plates, 3 bowls, 3 mugs, 3 glasses, and sets of utensils.",
        "Prepare the table by placing 3 large and 3 small plates, 3 bowls, 3 mugs, 3 glasses, and sets of utensils.",
        "Set up the dining table with 3 large and 3 small plates, along with 3 bowls, 3 mugs, 3 glasses, and sets of utensils."
    ],
    'Stack on table: 3 each large/small plates, bowls':[
        "Stack on the table: 3 each large/small plates, bowls.",
        "Create stacks on the table with 3 large and 3 small plates, as well as 3 bowls.",
        "Arrange 3 large and 3 small plates, along with 3 bowls, in neat stacks on the table.",
        "Form stacks on the table using 3 large and 3 small plates, and 3 bowls."
    ],
    'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils':[
        "Load the dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils.",
        "Place in the dishwasher 3 large and 3 small plates, 3 bowls, 3 mugs, 3 glasses, and sets of utensils.",
        "Load the dishwasher with 3 large and 3 small plates, 3 bowls, 3 mugs, 3 glasses, and sets of utensils.",
        "Arrange 3 large and 3 small plates, 3 bowls, 3 mugs, 3 glasses, and sets of utensils in the dishwasher for cleaning."
    ],
    'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils':[
        "Unload the dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils.",
        "Remove from the dishwasher 3 large and 3 small plates, 3 bowls, 3 mugs, 3 glasses, and sets of utensils.",
        "Empty the dishwasher of 3 large and 3 small plates, 3 bowls, 3 mugs, 3 glasses, and sets of utensils.",
        "Take out 3 large and 3 small plates, 3 bowls, 3 mugs, 3 glasses, and sets of utensils from the dishwasher."
    ],
}
