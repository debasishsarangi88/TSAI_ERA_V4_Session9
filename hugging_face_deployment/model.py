"""
ResNet50 Model Architecture for Hugging Face Deployment
This file contains the ResNet50 implementation that matches the training code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50(num_classes=1000):
    """Create ResNet50 model with specified number of classes"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def load_model_from_checkpoint(checkpoint_path, num_classes=1000, device='cpu'):
    """
    Load model from checkpoint file
    
    Args:
        checkpoint_path (str): Path to the .pth checkpoint file
        num_classes (int): Number of output classes
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        torch.nn.Module: Loaded model
    """
    model = ResNet50(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


# ImageNet class labels
IMAGENET_CLASSES = [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
    'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling',
    'goldfinch', 'house finch', 'junco', 'indigo bunting', 'robin',
    'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite',
    'bald eagle', 'vulture', 'great grey owl', 'European fire salamander',
    'common newt', 'eft', 'spotted salamander', 'axolotl', 'bullfrog',
    'tree frog', 'tailed frog', 'loggerhead', 'leatherback turtle',
    'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'common iguana',
    'American chameleon', 'whiptail', 'agama', 'frilled lizard',
    'alligator lizard', 'Gila monster', 'green lizard', 'African chameleon',
    'Komodo dragon', 'African crocodile', 'American alligator', 'trilobite',
    'harvestman', 'scorpion', 'black and gold garden spider', 'barn spider',
    'garden spider', 'black widow', 'tarantula', 'wolf spider', 'tick',
    'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie chicken',
    'peacock', 'quail', 'partridge', 'African grey', 'macaw', 'sulphur-crested cockatoo',
    'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar',
    'toucan', 'drake', 'red-breasted merganser', 'goose', 'black swan',
    'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat',
    'jellyfish', 'sea anemone', 'brain coral', 'flatworm', 'nematode',
    'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus',
    'Dungeness crab', 'rock crab', 'fiddler crab', 'king crab', 'American lobster',
    'spiny lobster', 'crayfish', 'hermit crab', 'isopod', 'white stork',
    'black stork', 'spoonbill', 'flamingo', 'little blue heron', 'American egret',
    'bittern', 'crane', 'limpkin', 'European gallinule', 'American coot',
    'bustard', 'ruddy turnstone', 'red-backed sandpiper', 'redshank', 'dowitcher',
    'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale',
    'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese spaniel',
    'Maltese dog', 'Pekinese', 'Shih-Tzu', 'Blenheim spaniel', 'papillon',
    'toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'basset', 'beagle',
    'bloodhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound',
    'English foxhound', 'redbone', 'borzoi', 'Irish wolfhound', 'Italian greyhound',
    'whippet', 'Ibizan hound', 'Norwegian elkhound', 'otterhound', 'Saluki',
    'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier',
    'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier',
    'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier',
    'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier',
    'Sealyham terrier', 'Airedale', 'cairn', 'Australian terrier', 'Dandie Dinmont',
    'Boston bull', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer',
    'Scotch terrier', 'Tibetan terrier', 'silky terrier', 'soft-coated wheaten terrier',
    'West Highland white terrier', 'Lhasa', 'flat-coated retriever', 'curly-coated retriever',
    'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever',
    'German short-haired pointer', 'vizsla', 'English setter', 'Irish setter',
    'Gordon setter', 'Brittany spaniel', 'clumber', 'English springer',
    'Welsh springer spaniel', 'cocker spaniel', 'Sussex spaniel', 'Irish water spaniel',
    'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie',
    'komondor', 'Old English sheepdog', 'Shetland sheepdog', 'collie',
    'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd',
    'Doberman', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog',
    'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff',
    'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'malamute',
    'Siberian husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg',
    'Newfoundland', 'Great Pyrenees', 'Samoyed', 'Pomeranian', 'chow',
    'keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'toy poodle',
    'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf',
    'white wolf', 'red wolf', 'coyote', 'dingo', 'dhole', 'African hunting dog',
    'hyena', 'red fox', 'kit fox', 'Arctic fox', 'grey fox', 'tabby',
    'tiger cat', 'Persian cat', 'Siamese cat', 'Egyptian Mau', 'cougar',
    'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah',
    'brown bear', 'American black bear', 'ice bear', 'sloth bear', 'mongoose',
    'meerkat', 'tiger beetle', 'ladybug', 'ground beetle', 'long-horned beetle',
    'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly',
    'bee', 'ant', 'grasshopper', 'cricket', 'walking stick', 'cockroach',
    'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly',
    'admiral', 'ringlet', 'monarch', 'cabbage butterfly', 'sulphur butterfly',
    'lycaenid', 'starfish', 'sea urchin', 'sea cucumber', 'wood rabbit',
    'hare', 'Angora', 'hamster', 'porcupine', 'fox squirrel', 'marmot',
    'beaver', 'guinea pig', 'sorrel', 'zebra', 'hog', 'wild boar',
    'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram',
    'bighorn', 'ibex', 'hartebeest', 'impala', 'gazelle', 'Arabian camel',
    'llama', 'weasel', 'mink', 'polecat', 'black-footed ferret', 'otter',
    'skunk', 'badger', 'armadillo', 'three-toed sloth', 'orangutan',
    'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas',
    'baboon', 'macaque', 'langur', 'colobus', 'proboscis monkey', 'marmoset',
    'capuchin', 'howler monkey', 'titi', 'spider monkey', 'squirrel monkey',
    'Madagascar cat', 'indri', 'Indian elephant', 'African elephant',
    'lesser panda', 'giant panda', 'barracouta', 'eel', 'coho', 'rock beauty',
    'anemone fish', 'sturgeon', 'gar', 'lionfish', 'puffer', 'abacus',
    'abaya', 'academic gown', 'accordion', 'acoustic guitar', 'aircraft carrier',
    'airliner', 'airship', 'altar', 'ambulance', 'amphibian', 'analog clock',
    'apiary', 'apron', 'ashcan', 'assault rifle', 'backpack', 'bakery',
    'balance beam', 'balloon', 'ballpoint', 'Band Aid', 'banjo', 'bannister',
    'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel',
    'barrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'bathing cap',
    'bath towel', 'bathtub', 'beach wagon', 'beacon', 'beaker', 'bearskin',
    'beer bottle', 'beer glass', 'bell cote', 'bib', 'bicycle-built-for-two',
    'bikini', 'binder', 'binoculars', 'birdhouse', 'boathouse', 'bobsled',
    'bolo tie', 'bonnet', 'bookcase', 'bookshop', 'bottlecap', 'bow',
    'bow tie', 'brass', 'brassiere', 'breakwater', 'breastplate', 'broom',
    'bucket', 'buckle', 'bulletproof vest', 'bullet train', 'butcher shop',
    'cab', 'caldron', 'candle', 'cannon', 'canoe', 'can opener', 'cardigan',
    'car mirror', 'carousel', "carpenter's kit", 'carton', 'car wheel',
    'cash machine', 'cassette', 'cassette player', 'castle', 'catamaran',
    'CD player', 'cello', 'cellular telephone', 'chain', 'chainlink fence',
    'chain mail', 'chain saw', 'chest', 'chiffonier', 'chime', 'china cabinet',
    'Christmas stocking', 'church', 'cinema', 'cleaver', 'cliff dwelling',
    'cloak', 'clog', 'cocktail shaker', 'coffee mug', 'coffeepot', 'coil',
    'combination lock', 'computer keyboard', 'confectionery', 'container ship',
    'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat', 'cradle',
    'crane', 'crash helmet', 'crate', 'crib', 'Crock Pot', 'croquet ball',
    'crutch', 'cuirass', 'dam', 'desk', 'desktop computer', 'dial telephone',
    'diaper', 'digital clock', 'digital watch', 'dining table', 'dishrag',
    'dishwasher', 'disk brake', 'dock', 'dogsled', 'dome', 'doormat',
    'drilling platform', 'drum', 'drumstick', 'dumbbell', 'Dutch oven',
    'electric fan', 'electric guitar', 'electric locomotive', 'entertainment center',
    'envelope', 'espresso maker', 'face powder', 'feather boa', 'file',
    'fireboat', 'fire engine', 'fire screen', 'flagpole', 'flute', 'folding chair',
    'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster',
    'freight car', 'French horn', 'frying pan', 'fur coat', 'garbage truck',
    'gasmask', 'gas pump', 'goblet', 'go-kart', 'golf ball', 'golfcart',
    'gondola', 'gong', 'gown', 'grand piano', 'greenhouse', 'grille',
    'grocery store', 'guillotine', 'hair slide', 'hair spray', 'half track',
    'hammer', 'hamper', 'hand blower', 'hand-held computer', 'handkerchief',
    'hard disc', 'harmonica', 'harp', 'harvester', 'hatchet', 'holster',
    'home theater', 'honeycomb', 'hook', 'hoopskirt', 'horizontal bar',
    'horse cart', 'hourglass', 'iPod', 'iron', "jack-o'-lantern", 'jean',
    'jeep', 'jersey', 'jigsaw puzzle', 'jinrikisha', 'joystick', 'kimono',
    'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade', 'laptop',
    'lawn mower', 'lens cap', 'letter opener', 'library', 'lifeboat',
    'lighter', 'limousine', 'liner', 'lipstick', 'Loafer', 'lotion',
    'loudspeaker', 'loupe', 'lumbermill', 'magnetic compass', 'mailbag',
    'mailbox', 'maillot', 'manhole cover', 'maraca', 'marimba', 'mask',
    'matchstick', 'maypole', 'maze', 'measuring cup', 'medicine chest',
    'megalith', 'microphone', 'microwave', 'military uniform', 'milk can',
    'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl',
    'mobile home', 'Model T', 'modem', 'monastery', 'monitor', 'moped',
    'mortar', 'mortarboard', 'mosque', 'mosquito net', 'motor scooter',
    'mountain bike', 'mountain tent', 'mouse', 'mousetrap', 'moving van',
    'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook',
    'obelisk', 'oboe', 'ocarina', 'odometer', 'oil filter', 'organ',
    'oscilloscope', 'overskirt', 'oxcart', 'oxygen mask', 'packet',
    'paddle', 'paddlewheel', 'padlock', 'paintbrush', 'pajama', 'palace',
    'panpipe', 'paper towel', 'parachute', 'parallel bars', 'park bench',
    'parking meter', 'passenger car', 'patio', 'pay-phone', 'pedestal',
    'pencil box', 'pencil sharpener', 'perfume', 'Petri dish', 'photocopier',
    'pick', 'pickelhaube', 'picket fence', 'pickup', 'pier', 'piggy bank',
    'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate',
    'pitcher', 'plane', 'planetarium', 'plastic bag', 'plate rack',
    'plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho',
    'pool table', 'pop bottle', 'pot', "potter's wheel", 'power drill',
    'prayer rug', 'printer', 'prison', 'puck', 'punching bag', 'purse',
    'quill', 'quilt', 'racer', 'racket', 'radiator', 'radio', 'radio telescope',
    'rain barrel', 'ramjet', 'red wine', 'reel', 'reflex camera', 'refrigerator',
    'remote control', 'restaurant', 'revolver', 'rifle', 'rocking chair',
    'rotisserie', 'rubber eraser', 'rugby ball', 'rule', 'running shoe',
    'safe', 'safety pin', 'saltshaker', 'sandal', 'sarong', 'sax',
    'scabbard', 'scale', 'school bus', 'schooner', 'scoreboard', 'screen',
    'screw', 'screwdriver', 'seat belt', 'sewing machine', 'shield',
    'shoe shop', 'shoji', 'shopping basket', 'shopping cart', 'shovel',
    'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag',
    'slide rule', 'sliding door', 'slot', 'snorkel', 'snowmobile', 'snowplow',
    'soap dispenser', 'soccer ball', 'sock', 'solar dish', 'sombrero',
    'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula',
    'speedboat', 'spider web', 'spindle', 'sports car', 'spotlight',
    'stage', 'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope',
    'stole', 'stone wall', 'stopwatch', 'stove', 'strainer', 'streetcar',
    'stretcher', 'studio couch', 'stupa', 'submarine', 'suit', 'sundial',
    'sunglass', 'sunglasses', 'sunscreen', 'suspension bridge', 'swab',
    'sweatshirt', 'swimming trunks', 'swing', 'switch', 'syringe',
    'table lamp', 'tank', 'tape player', 'teapot', 'teddy', 'television',
    'tennis ball', 'thatch', 'theater curtain', 'thimble', 'thresher',
    'throne', 'tile roof', 'toaster', 'tobacco shop', 'toilet seat',
    'torch', 'totem pole', 'tow truck', 'toyshop', 'tractor', 'trailer truck',
    'tray', 'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch',
    'trolleybus', 'trombone', 'tub', 'turnstile', 'typewriter keyboard',
    'umbrella', 'unicycle', 'upright', 'vacuum', 'vase', 'vault',
    'velvet', 'vending machine', 'vestment', 'viaduct', 'violin', 'virgin',
    'volleyball', 'waffle iron', 'wall clock', 'wallet', 'wardrobe',
    'warplane', 'washbasin', 'washer', 'water bottle', 'water jug',
    'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen',
    'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok',
    'wooden spoon', 'wool', 'worm fence', 'wreck', 'yawl', 'yurt',
    'web site', 'comic book', 'crossword puzzle', 'street sign', 'traffic light',
    'book jacket', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot',
    'trifle', 'ice cream', 'ice lolly', 'French loaf', 'bagel', 'pretzel',
    'cheeseburger', 'hotdog', 'mashed potato', 'head cabbage', 'broccoli',
    'cauliflower', 'zucchini', 'spaghetti squash', 'acorn squash', 'butternut squash',
    'cucumber', 'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith',
    'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit',
    'custard apple', 'pomegranate', 'hay', 'carbonara', 'chocolate sauce',
    'dough', 'meat loaf', 'pizza', 'potpie', 'burrito', 'red wine', 'espresso',
    'cup', 'eggnog', 'alp', 'bubble', 'cliff', 'coral reef', 'geyser',
    'lakeside', 'promontory', 'sandbar', 'seashore', 'valley', 'volcano',
    'ballplayer', 'groom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper",
    'corn', 'acorn', 'hip', 'buckeye', 'coral fungus', 'agaric', 'gyromitra',
    'stinkhorn', 'earthstar', 'hen-of-the-woods', 'bolete', 'ear', 'toilet tissue'
]
