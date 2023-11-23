import torch
from torch.utils.data import Dataset
import torchvision
import os
import pickle
import numpy as np
import json
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import scipy.io as sio
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
from robustness import datasets

class ETHECDataset(Dataset):
    def __init__(self, root_dir, split, transform):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        if self.split == 'train':
            data_file = os.path.join(self.root_dir, 'splits', f'{self.split}_clean.json')
        else:
            data_file = os.path.join(self.root_dir, 'splits', f'{self.split}.json')
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_info  = self.data[list(self.data.keys())[idx]]
        image_name = image_info['image_name']
        image_path = os.path.join(self.root_dir, 'IMAGO_build_test_resized', image_info['image_path'])

        image = Image.open(os.path.join(image_path, image_name)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        labels_family = image_info['family']
        labels_subfamily = image_info['subfamily']
        labels_genus = image_info['genus']
        labels_specific_epithet = image_info['specific_epithet']

        return image, labels_family, labels_subfamily, labels_genus, labels_specific_epithet

class ETHECLabelMap:
    """
    Implements map from labels to hot vectors for ETHEC database.
    https://github.com/ankitdhall/learning_embeddings/blob/master/data/db.py
    """

    def __init__(self):
        self.family = {
            "Hesperiidae": 0,
            "Papilionidae": 1,
            "Pieridae": 2,
            "Nymphalidae": 3,
            "Lycaenidae": 4,
            "Riodinidae": 5
        }
        self.subfamily = {
            "Heteropterinae": 0,
            "Hesperiinae": 1,
            "Pyrginae": 2,
            "Parnassiinae": 3,
            "Papilioninae": 4,
            "Dismorphiinae": 5,
            "Coliadinae": 6,
            "Pierinae": 7,
            "Satyrinae": 8,
            "Lycaeninae": 9,
            "Nymphalinae": 10,
            "Heliconiinae": 11,
            "Nemeobiinae": 12,
            "Theclinae": 13,
            "Aphnaeinae": 14,
            "Polyommatinae": 15,
            "Libytheinae": 16,
            "Danainae": 17,
            "Charaxinae": 18,
            "Apaturinae": 19,
            "Limenitidinae": 20
        }

        self.genus = {
            "Carterocephalus": 0,
            "Heteropterus": 1,
            "Thymelicus": 2,
            "Hesperia": 3,
            "Ochlodes": 4,
            "Gegenes": 5,
            "Erynnis": 6,
            "Carcharodus": 7,
            "Spialia": 8,
            "Muschampia": 9,
            "Pyrgus": 10,
            "Parnassius": 11,
            "Archon": 12,
            "Sericinus": 13,
            "Zerynthia": 14,
            "Allancastria": 15,
            "Bhutanitis": 16,
            "Luehdorfia": 17,
            "Papilio": 18,
            "Iphiclides": 19,
            "Leptidea": 20,
            "Colias": 21,
            "Aporia": 22,
            "Catopsilia": 23,
            "Gonepteryx": 24,
            "Mesapia": 25,
            "Baltia": 26,
            "Pieris": 27,
            "Erebia": 28,
            "Berberia": 29,
            "Proterebia": 30,
            "Boeberia": 31,
            "Loxerebia": 32,
            "Lycaena": 33,
            "Melitaea": 34,
            "Argynnis": 35,
            "Heliophorus": 36,
            "Cethosia": 37,
            "Childrena": 38,
            "Pontia": 39,
            "Anthocharis": 40,
            "Zegris": 41,
            "Euchloe": 42,
            "Colotis": 43,
            "Hamearis": 44,
            "Polycaena": 45,
            "Favonius": 46,
            "Cigaritis": 47,
            "Tomares": 48,
            "Chrysozephyrus": 49,
            "Ussuriana": 50,
            "Coreana": 51,
            "Japonica": 52,
            "Thecla": 53,
            "Celastrina": 54,
            "Laeosopis": 55,
            "Callophrys": 56,
            "Zizeeria": 57,
            "Tarucus": 58,
            "Cyclyrius": 59,
            "Leptotes": 60,
            "Satyrium": 61,
            "Lampides": 62,
            "Neolycaena": 63,
            "Cupido": 64,
            "Maculinea": 65,
            "Glaucopsyche": 66,
            "Pseudophilotes": 67,
            "Scolitantides": 68,
            "Iolana": 69,
            "Plebejus": 70,
            "Agriades": 71,
            "Plebejidea": 72,
            "Kretania": 73,
            "Aricia": 74,
            "Pamiria": 75,
            "Polyommatus": 76,
            "Eumedonia": 77,
            "Cyaniris": 78,
            "Lysandra": 79,
            "Glabroculus": 80,
            "Neolysandra": 81,
            "Libythea": 82,
            "Danaus": 83,
            "Charaxes": 84,
            "Apatura": 85,
            "Limenitis": 86,
            "Euapatura": 87,
            "Hestina": 88,
            "Timelaea": 89,
            "Mimathyma": 90,
            "Lelecella": 91,
            "Neptis": 92,
            "Nymphalis": 93,
            "Inachis": 94,
            "Araschnia": 95,
            "Vanessa": 96,
            "Speyeria": 97,
            "Fabriciana": 98,
            "Argyronome": 99,
            "Issoria": 100,
            "Brenthis": 101,
            "Boloria": 102,
            "Kuekenthaliella": 103,
            "Clossiana": 104,
            "Proclossiana": 105,
            "Euphydryas": 106,
            "Melanargia": 107,
            "Davidina": 108,
            "Hipparchia": 109,
            "Chazara": 110,
            "Pseudochazara": 111,
            "Karanasa": 112,
            "Oeneis": 113,
            "Satyrus": 114,
            "Minois": 115,
            "Arethusana": 116,
            "Brintesia": 117,
            "Maniola": 118,
            "Aphantopus": 119,
            "Hyponephele": 120,
            "Pyronia": 121,
            "Coenonympha": 122,
            "Pararge": 123,
            "Ypthima": 124,
            "Lasiommata": 125,
            "Lopinga": 126,
            "Kirinia": 127,
            "Neope": 128,
            "Atrophaneura": 129,
            "Agehana": 130,
            "Arisbe": 131,
            "Teinopalpus": 132,
            "Graphium": 133,
            "Meandrusa": 134
        }

        self.genus_specific_epithet = {
            "Carterocephalus_palaemon": 0,
            "Heteropterus_morpheus": 1,
            "Thymelicus_sylvestris": 2,
            "Thymelicus_lineola": 3,
            "Thymelicus_acteon": 4,
            "Hesperia_comma": 5,
            "Ochlodes_venata": 6,
            "Gegenes_nostrodamus": 7,
            "Erynnis_tages": 8,
            "Carcharodus_alceae": 9,
            "Carcharodus_lavatherae": 10,
            "Carcharodus_baeticus": 11,
            "Carcharodus_floccifera": 12,
            "Spialia_sertorius": 13,
            "Spialia_orbifer": 14,
            "Muschampia_proto": 15,
            "Pyrgus_alveus": 16,
            "Pyrgus_armoricanus": 17,
            "Pyrgus_andromedae": 18,
            "Pyrgus_cacaliae": 19,
            "Pyrgus_carlinae": 20,
            "Pyrgus_carthami": 21,
            "Pyrgus_malvae": 22,
            "Pyrgus_cinarae": 23,
            "Pyrgus_cirsii": 24,
            "Pyrgus_malvoides": 25,
            "Pyrgus_onopordi": 26,
            "Pyrgus_serratulae": 27,
            "Pyrgus_sidae": 28,
            "Pyrgus_warrenensis": 29,
            "Parnassius_sacerdos": 30,
            "Archon_apollinus": 31,
            "Parnassius_apollo": 32,
            "Parnassius_mnemosyne": 33,
            "Parnassius_glacialis": 34,
            "Sericinus_montela": 35,
            "Zerynthia_rumina": 36,
            "Zerynthia_polyxena": 37,
            "Allancastria_cerisyi": 38,
            "Allancastria_deyrollei": 39,
            "Allancastria_caucasica": 40,
            "Bhutanitis_thaidina": 41,
            "Bhutanitis_lidderdalii": 42,
            "Bhutanitis_mansfieldi": 43,
            "Luehdorfia_japonica": 44,
            "Luehdorfia_puziloi": 45,
            "Luehdorfia_chinensis": 46,
            "Papilio_machaon": 47,
            "Parnassius_stubbendorfii": 48,
            "Parnassius_apollonius": 49,
            "Papilio_alexanor": 50,
            "Papilio_hospiton": 51,
            "Papilio_xuthus": 52,
            "Iphiclides_podalirius": 53,
            "Iphiclides_feisthamelii": 54,
            "Leptidea_sinapis": 55,
            "Colias_palaeno": 56,
            "Colias_pelidne": 57,
            "Leptidea_juvernica": 58,
            "Leptidea_morsei": 59,
            "Leptidea_amurensis": 60,
            "Leptidea_duponcheli": 61,
            "Colias_marcopolo": 62,
            "Colias_ladakensis": 63,
            "Colias_nebulosa": 64,
            "Colias_nastes": 65,
            "Colias_cocandica": 66,
            "Colias_sieversi": 67,
            "Colias_sifanica": 68,
            "Colias_alpherakii": 69,
            "Colias_christophi": 70,
            "Colias_tyche": 71,
            "Colias_phicomone": 72,
            "Colias_alfacariensis": 73,
            "Colias_hyale": 74,
            "Colias_erate": 75,
            "Colias_erschoffi": 76,
            "Colias_romanovi": 77,
            "Colias_regia": 78,
            "Colias_stoliczkana": 79,
            "Colias_hecla": 80,
            "Colias_eogene": 81,
            "Colias_thisoa": 82,
            "Colias_staudingeri": 83,
            "Colias_lada": 84,
            "Colias_baeckeri": 85,
            "Colias_fieldii": 86,
            "Colias_heos": 87,
            "Colias_caucasica": 88,
            "Colias_diva": 89,
            "Colias_chrysotheme": 90,
            "Colias_balcanica": 91,
            "Colias_myrmidone": 92,
            "Colias_croceus": 93,
            "Colias_felderi": 94,
            "Colias_viluiensis": 95,
            "Aporia_crataegi": 96,
            "Colias_aurorina": 97,
            "Colias_chlorocoma": 98,
            "Colias_libanotica": 99,
            "Colias_wiskotti": 100,
            "Catopsilia_florella": 101,
            "Gonepteryx_rhamni": 102,
            "Gonepteryx_maxima": 103,
            "Gonepteryx_cleopatra": 104,
            "Gonepteryx_cleobule": 105,
            "Gonepteryx_amintha": 106,
            "Aporia_procris": 107,
            "Mesapia_peloria": 108,
            "Aporia_potanini": 109,
            "Aporia_nabellica": 110,
            "Baltia_butleri": 111,
            "Pieris_brassicae": 112,
            "Pieris_cheiranthi": 113,
            "Pieris_rapae": 114,
            "Erebia_gorge": 115,
            "Erebia_aethiopellus": 116,
            "Erebia_mnestra": 117,
            "Erebia_epistygne": 118,
            "Erebia_ottomana": 119,
            "Erebia_tyndarus": 120,
            "Erebia_oeme": 121,
            "Erebia_lefebvrei": 122,
            "Erebia_melas": 123,
            "Erebia_zapateri": 124,
            "Erebia_neoridas": 125,
            "Erebia_montana": 126,
            "Erebia_cassioides": 127,
            "Erebia_nivalis": 128,
            "Erebia_scipio": 129,
            "Erebia_pronoe": 130,
            "Erebia_styx": 131,
            "Erebia_meolans": 132,
            "Erebia_palarica": 133,
            "Erebia_pandrose": 134,
            "Erebia_meta": 135,
            "Erebia_erinnyn": 136,
            "Berberia_lambessanus": 137,
            "Berberia_abdelkader": 138,
            "Proterebia_afra": 139,
            "Boeberia_parmenio": 140,
            "Loxerebia_saxicola": 141,
            "Pieris_mannii": 142,
            "Pieris_ergane": 143,
            "Pieris_krueperi": 144,
            "Pieris_napi": 145,
            "Lycaena_thersamon": 146,
            "Lycaena_lampon": 147,
            "Lycaena_solskyi": 148,
            "Lycaena_splendens": 149,
            "Lycaena_candens": 150,
            "Lycaena_ochimus": 151,
            "Lycaena_hippothoe": 152,
            "Lycaena_tityrus": 153,
            "Lycaena_thetis": 154,
            "Melitaea_athalia": 155,
            "Argynnis_paphia": 156,
            "Heliophorus_tamu": 157,
            "Heliophorus_brahma": 158,
            "Heliophorus_androcles": 159,
            "Cethosia_biblis": 160,
            "Childrena_childreni": 161,
            "Melitaea_parthenoides": 162,
            "Pieris_bryoniae": 163,
            "Pontia_edusa": 164,
            "Pontia_daplidice": 165,
            "Pontia_callidice": 166,
            "Anthocharis_thibetana": 167,
            "Anthocharis_bambusarum": 168,
            "Anthocharis_bieti": 169,
            "Anthocharis_scolymus": 170,
            "Zegris_pyrothoe": 171,
            "Zegris_eupheme": 172,
            "Zegris_fausti": 173,
            "Euchloe_simplonia": 174,
            "Pontia_chloridice": 175,
            "Euchloe_belemia": 176,
            "Euchloe_ausonia": 177,
            "Euchloe_tagis": 178,
            "Euchloe_crameri": 179,
            "Euchloe_insularis": 180,
            "Euchloe_orientalis": 181,
            "Euchloe_transcaspica": 182,
            "Euchloe_charlonia": 183,
            "Euchloe_tomyris": 184,
            "Anthocharis_gruneri": 185,
            "Anthocharis_damone": 186,
            "Anthocharis_cardamines": 187,
            "Anthocharis_belia": 188,
            "Anthocharis_euphenoides": 189,
            "Colotis_fausta": 190,
            "Colotis_evagore": 191,
            "Hamearis_lucina": 192,
            "Polycaena_tamerlana": 193,
            "Lycaena_phlaeas": 194,
            "Lycaena_helle": 195,
            "Lycaena_pang": 196,
            "Lycaena_caspius": 197,
            "Lycaena_margelanica": 198,
            "Lycaena_dispar": 199,
            "Lycaena_alciphron": 200,
            "Lycaena_virgaureae": 201,
            "Lycaena_kasyapa": 202,
            "Favonius_quercus": 203,
            "Cigaritis_siphax": 204,
            "Cigaritis_allardi": 205,
            "Tomares_ballus": 206,
            "Tomares_nogelii": 207,
            "Tomares_mauretanicus": 208,
            "Tomares_romanovi": 209,
            "Tomares_callimachus": 210,
            "Chrysozephyrus_smaragdinus": 211,
            "Ussuriana_micahaelis": 212,
            "Coreana_raphaelis": 213,
            "Japonica_saepestriata": 214,
            "Thecla_betulae": 215,
            "Celastrina_argiolus": 216,
            "Laeosopis_roboris": 217,
            "Callophrys_rubi": 218,
            "Zizeeria_knysna": 219,
            "Tarucus_theophrastus": 220,
            "Cyclyrius_webbianus": 221,
            "Tarucus_balkanica": 222,
            "Leptotes_pirithous": 223,
            "Satyrium_spini": 224,
            "Lampides_boeticus": 225,
            "Satyrium_w-album": 226,
            "Satyrium_ilicis": 227,
            "Satyrium_pruni": 228,
            "Satyrium_acaciae": 229,
            "Satyrium_esculi": 230,
            "Neolycaena_rhymnus": 231,
            "Callophrys_avis": 232,
            "Cupido_minimus": 233,
            "Maculinea_rebeli": 234,
            "Maculinea_arion": 235,
            "Cupido_alcetas": 236,
            "Cupido_osiris": 237,
            "Cupido_argiades": 238,
            "Cupido_decolorata": 239,
            "Glaucopsyche_melanops": 240,
            "Glaucopsyche_alexis": 241,
            "Maculinea_alcon": 242,
            "Maculinea_teleius": 243,
            "Pseudophilotes_abencerragus": 244,
            "Pseudophilotes_panoptes": 245,
            "Pseudophilotes_vicrama": 246,
            "Pseudophilotes_baton": 247,
            "Maculinea_nausithous": 248,
            "Scolitantides_orion": 249,
            "Iolana_gigantea": 250,
            "Iolana_iolas": 251,
            "Plebejus_argus": 252,
            "Plebejus_eversmanni": 253,
            "Glaucopsyche_paphos": 254,
            "Plebejus_argyrognomon": 255,
            "Agriades_optilete": 256,
            "Plebejidea_loewii": 257,
            "Plebejus_idas": 258,
            "Kretania_trappi": 259,
            "Kretania_pylaon": 260,
            "Kretania_martini": 261,
            "Plebejus_samudra": 262,
            "Agriades_orbitulus": 263,
            "Aricia_artaxerxes": 264,
            "Pamiria_omphisa": 265,
            "Agriades_glandon": 266,
            "Aricia_agestis": 267,
            "Polyommatus_damon": 268,
            "Eumedonia_eumedon": 269,
            "Aricia_nicias": 270,
            "Cyaniris_semiargus": 271,
            "Polyommatus_dolus": 272,
            "Aricia_anteros": 273,
            "Polyommatus_antidolus": 274,
            "Polyommatus_phyllis": 275,
            "Polyommatus_iphidamon": 276,
            "Polyommatus_damonides": 277,
            "Polyommatus_damone": 278,
            "Polyommatus_ripartii": 279,
            "Polyommatus_admetus": 280,
            "Polyommatus_dorylas": 281,
            "Polyommatus_erschoffi": 282,
            "Polyommatus_thersites": 283,
            "Polyommatus_escheri": 284,
            "Lysandra_bellargus": 285,
            "Lysandra_coridon": 286,
            "Lysandra_hispana": 287,
            "Lysandra_albicans": 288,
            "Lysandra_caelestissima": 289,
            "Lysandra_punctifera": 290,
            "Polyommatus_nivescens": 291,
            "Polyommatus_aedon": 292,
            "Polyommatus_atys": 293,
            "Polyommatus_icarus": 294,
            "Polyommatus_caeruleus": 295,
            "Glabroculus_elvira": 296,
            "Glabroculus_cyane": 297,
            "Polyommatus_stoliczkana": 298,
            "Polyommatus_golgus": 299,
            "Neolysandra_coelestina": 300,
            "Neolysandra_corona": 301,
            "Polyommatus_amandus": 302,
            "Polyommatus_daphnis": 303,
            "Polyommatus_eros": 304,
            "Polyommatus_celina": 305,
            "Libythea_celtis": 306,
            "Danaus_plexippus": 307,
            "Danaus_chrysippus": 308,
            "Charaxes_jasius": 309,
            "Apatura_iris": 310,
            "Apatura_ilia": 311,
            "Limenitis_reducta": 312,
            "Apatura_metis": 313,
            "Euapatura_mirza": 314,
            "Hestina_japonica": 315,
            "Timelaea_albescens": 316,
            "Limenitis_populi": 317,
            "Limenitis_camilla": 318,
            "Mimathyma_schrenckii": 319,
            "Limenitis_sydyi": 320,
            "Lelecella_limenitoides": 321,
            "Neptis_sappho": 322,
            "Neptis_rivularis": 323,
            "Nymphalis_antiopa": 324,
            "Nymphalis_polychloros": 325,
            "Nymphalis_xanthomelas": 326,
            "Nymphalis_l-album": 327,
            "Nymphalis_urticae": 328,
            "Nymphalis_ichnusa": 329,
            "Nymphalis_egea": 330,
            "Nymphalis_c-album": 331,
            "Inachis_io": 332,
            "Araschnia_burejana": 333,
            "Araschnia_levana": 334,
            "Nymphalis_canace": 335,
            "Nymphalis_c-aureum": 336,
            "Vanessa_atalanta": 337,
            "Vanessa_vulcania": 338,
            "Vanessa_cardui": 339,
            "Argynnis_pandora": 340,
            "Speyeria_aglaja": 341,
            "Fabriciana_niobe": 342,
            "Speyeria_clara": 343,
            "Argyronome_laodice": 344,
            "Fabriciana_adippe": 345,
            "Fabriciana_jainadeva": 346,
            "Fabriciana_auresiana": 347,
            "Fabriciana_elisa": 348,
            "Issoria_lathonia": 349,
            "Brenthis_hecate": 350,
            "Brenthis_daphne": 351,
            "Brenthis_ino": 352,
            "Boloria_pales": 353,
            "Kuekenthaliella_eugenia": 354,
            "Boloria_aquilonaris": 355,
            "Boloria_napaea": 356,
            "Clossiana_selene": 357,
            "Proclossiana_eunomia": 358,
            "Boloria_graeca": 359,
            "Clossiana_thore": 360,
            "Clossiana_dia": 361,
            "Clossiana_euphrosyne": 362,
            "Clossiana_titania": 363,
            "Clossiana_freija": 364,
            "Melitaea_cinxia": 365,
            "Melitaea_phoebe": 366,
            "Melitaea_didyma": 367,
            "Melitaea_varia": 368,
            "Melitaea_aurelia": 369,
            "Melitaea_asteria": 370,
            "Melitaea_diamina": 371,
            "Melitaea_britomartis": 372,
            "Melitaea_acraeina": 373,
            "Melitaea_trivia": 374,
            "Melitaea_persea": 375,
            "Melitaea_ambigua": 376,
            "Melitaea_deione": 377,
            "Melitaea_turanica": 378,
            "Euphydryas_maturna": 379,
            "Euphydryas_ichnea": 380,
            "Euphydryas_cynthia": 381,
            "Euphydryas_aurinia": 382,
            "Euphydryas_sibirica": 383,
            "Euphydryas_iduna": 384,
            "Melanargia_titea": 385,
            "Melanargia_parce": 386,
            "Melanargia_lachesis": 387,
            "Melanargia_galathea": 388,
            "Melanargia_russiae": 389,
            "Melanargia_larissa": 390,
            "Melanargia_ines": 391,
            "Melanargia_pherusa": 392,
            "Melanargia_occitanica": 393,
            "Melanargia_arge": 394,
            "Melanargia_meridionalis": 395,
            "Melanargia_leda": 396,
            "Melanargia_halimede": 397,
            "Davidina_armandi": 398,
            "Hipparchia_semele": 399,
            "Chazara_briseis": 400,
            "Hipparchia_parisatis": 401,
            "Hipparchia_fidia": 402,
            "Hipparchia_genava": 403,
            "Hipparchia_aristaeus": 404,
            "Hipparchia_fagi": 405,
            "Hipparchia_wyssii": 406,
            "Hipparchia_fatua": 407,
            "Hipparchia_statilinus": 408,
            "Hipparchia_syriaca": 409,
            "Hipparchia_neomiris": 410,
            "Hipparchia_azorina": 411,
            "Chazara_prieuri": 412,
            "Chazara_bischoffii": 413,
            "Chazara_persephone": 414,
            "Pseudochazara_pelopea": 415,
            "Pseudochazara_beroe": 416,
            "Pseudochazara_schahrudensis": 417,
            "Pseudochazara_telephassa": 418,
            "Pseudochazara_anthelea": 419,
            "Pseudochazara_amalthea": 420,
            "Pseudochazara_graeca": 421,
            "Pseudochazara_cingovskii": 422,
            "Karanasa_modesta": 423,
            "Oeneis_magna": 424,
            "Oeneis_glacialis": 425,
            "Satyrus_actaea": 426,
            "Satyrus_parthicus": 427,
            "Satyrus_ferula": 428,
            "Minois_dryas": 429,
            "Arethusana_arethusa": 430,
            "Brintesia_circe": 431,
            "Maniola_jurtina": 432,
            "Aphantopus_hyperantus": 433,
            "Hyponephele_pulchra": 434,
            "Hyponephele_pulchella": 435,
            "Hyponephele_cadusia": 436,
            "Hyponephele_amardaea": 437,
            "Hyponephele_lycaon": 438,
            "Maniola_nurag": 439,
            "Hyponephele_lupina": 440,
            "Pyronia_tithonus": 441,
            "Coenonympha_gardetta": 442,
            "Coenonympha_tullia": 443,
            "Pyronia_bathseba": 444,
            "Pyronia_cecilia": 445,
            "Coenonympha_corinna": 446,
            "Coenonympha_pamphilus": 447,
            "Pyronia_janiroides": 448,
            "Coenonympha_dorus": 449,
            "Coenonympha_darwiniana": 450,
            "Coenonympha_arcania": 451,
            "Pararge_aegeria": 452,
            "Coenonympha_leander": 453,
            "Ypthima_baldus": 454,
            "Coenonympha_iphioides": 455,
            "Coenonympha_glycerion": 456,
            "Coenonympha_hero": 457,
            "Coenonympha_oedippus": 458,
            "Pararge_xiphioides": 459,
            "Lasiommata_megera": 460,
            "Lasiommata_petropolitana": 461,
            "Lasiommata_maera": 462,
            "Lasiommata_paramegaera": 463,
            "Lopinga_achine": 464,
            "Erebia_euryale": 465,
            "Kirinia_roxelana": 466,
            "Kirinia_climene": 467,
            "Neope_goschkevitschii": 468,
            "Erebia_ligea": 469,
            "Kirinia_eversmanni": 470,
            "Erebia_eriphyle": 471,
            "Erebia_manto": 472,
            "Erebia_epiphron": 473,
            "Erebia_flavofasciata": 474,
            "Erebia_bubastis": 475,
            "Erebia_claudina": 476,
            "Erebia_christi": 477,
            "Erebia_pharte": 478,
            "Erebia_aethiops": 479,
            "Erebia_melampus": 480,
            "Erebia_sudetica": 481,
            "Erebia_neriene": 482,
            "Erebia_triaria": 483,
            "Erebia_medusa": 484,
            "Erebia_alberganus": 485,
            "Erebia_pluto": 486,
            "Gonepteryx_farinosa": 487,
            "Melitaea_nevadensis": 488,
            "Agriades_pheretiades": 489,
            "Parnassius_eversmannii": 490,
            "Parnassius_ariadne": 491,
            "Parnassius_stenosemus": 492,
            "Parnassius_hardwickii": 493,
            "Parnassius_charltonius": 494,
            "Parnassius_imperator": 495,
            "Parnassius_acdestis": 496,
            "Parnassius_cardinal": 497,
            "Parnassius_szechenyii": 498,
            "Parnassius_delphius": 499,
            "Parnassius_maximinus": 500,
            "Parnassius_staudingeri": 501,
            "Parnassius_orleans": 502,
            "Parnassius_augustus": 503,
            "Parnassius_loxias": 504,
            "Parnassius_charltontonius": 505,
            "Parnassius_autocrator": 506,
            "Parnassius_stoliczkanus": 507,
            "Parnassius_nordmanni": 508,
            "Parnassius_simo": 509,
            "Parnassius_bremeri": 510,
            "Parnassius_actius": 511,
            "Parnassius_cephalus": 512,
            "Parnassius_maharaja": 513,
            "Parnassius_tenedius": 514,
            "Parnassius_acco": 515,
            "Parnassius_boedromius": 516,
            "Parnassius_tianschanicus": 517,
            "Parnassius_phoebus": 518,
            "Parnassius_honrathi": 519,
            "Parnassius_ruckbeili": 520,
            "Parnassius_epaphus": 521,
            "Parnassius_nomion": 522,
            "Parnassius_jacquemonti": 523,
            "Parnassius_mercurius": 524,
            "Parnassius_tibetanus": 525,
            "Parnassius_clodius": 526,
            "Parnassius_smintheus": 527,
            "Parnassius_behrii": 528,
            "Atrophaneura_mencius": 529,
            "Atrophaneura_plutonius": 530,
            "Papilio_dehaani": 531,
            "Papilio_polytes": 532,
            "Atrophaneura_horishana": 533,
            "Papilio_bootes": 534,
            "Agehana_elwesi": 535,
            "Papilio_maackii": 536,
            "Atrophaneura_impediens": 537,
            "Atrophaneura_polyeuctes": 538,
            "Arisbe_mandarinus": 539,
            "Arisbe_parus": 540,
            "Atrophaneura_alcinous": 541,
            "Arisbe_alebion": 542,
            "Papilio_helenus": 543,
            "Teinopalpus_imperialis": 544,
            "Arisbe_eurous": 545,
            "Graphium_sarpedon": 546,
            "Arisbe_doson": 547,
            "Arisbe_tamerlanus": 548,
            "Papilio_bianor": 549,
            "Papilio_paris": 550,
            "Atrophaneura_nevilli": 551,
            "Papilio_krishna": 552,
            "Papilio_macilentus": 553,
            "Arisbe_leechi": 554,
            "Papilio_protenor": 555,
            "Graphium_cloanthus": 556,
            "Papilio_castor": 557,
            "Meandrusa_sciron": 558,
            "Papilio_arcturus": 559,
            "Agriades_lehanus": 560
        }

        self.child_of_family = {
            "Hesperiidae": [
                "Heteropterinae",
                "Hesperiinae",
                "Pyrginae"
            ],
            "Papilionidae": [
                "Parnassiinae",
                "Papilioninae"
            ],
            "Pieridae": [
                "Dismorphiinae",
                "Coliadinae",
                "Pierinae"
            ],
            "Nymphalidae": [
                "Satyrinae",
                "Nymphalinae",
                "Heliconiinae",
                "Libytheinae",
                "Danainae",
                "Charaxinae",
                "Apaturinae",
                "Limenitidinae"
            ],
            "Lycaenidae": [
                "Lycaeninae",
                "Theclinae",
                "Aphnaeinae",
                "Polyommatinae"
            ],
            "Riodinidae": [
                "Nemeobiinae"
            ]
        }

        self.child_of_subfamily = {
            "Heteropterinae": [
                "Carterocephalus",
                "Heteropterus"
            ],
            "Hesperiinae": [
                "Thymelicus",
                "Hesperia",
                "Ochlodes",
                "Gegenes"
            ],
            "Pyrginae": [
                "Erynnis",
                "Carcharodus",
                "Spialia",
                "Muschampia",
                "Pyrgus"
            ],
            "Parnassiinae": [
                "Parnassius",
                "Archon",
                "Sericinus",
                "Zerynthia",
                "Allancastria",
                "Bhutanitis",
                "Luehdorfia"
            ],
            "Papilioninae": [
                "Papilio",
                "Iphiclides",
                "Atrophaneura",
                "Agehana",
                "Arisbe",
                "Teinopalpus",
                "Graphium",
                "Meandrusa"
            ],
            "Dismorphiinae": [
                "Leptidea"
            ],
            "Coliadinae": [
                "Colias",
                "Catopsilia",
                "Gonepteryx"
            ],
            "Pierinae": [
                "Aporia",
                "Mesapia",
                "Baltia",
                "Pieris",
                "Pontia",
                "Anthocharis",
                "Zegris",
                "Euchloe",
                "Colotis"
            ],
            "Satyrinae": [
                "Erebia",
                "Berberia",
                "Proterebia",
                "Boeberia",
                "Loxerebia",
                "Melanargia",
                "Davidina",
                "Hipparchia",
                "Chazara",
                "Pseudochazara",
                "Karanasa",
                "Oeneis",
                "Satyrus",
                "Minois",
                "Arethusana",
                "Brintesia",
                "Maniola",
                "Aphantopus",
                "Hyponephele",
                "Pyronia",
                "Coenonympha",
                "Pararge",
                "Ypthima",
                "Lasiommata",
                "Lopinga",
                "Kirinia",
                "Neope"
            ],
            "Lycaeninae": [
                "Lycaena",
                "Heliophorus"
            ],
            "Nymphalinae": [
                "Melitaea",
                "Nymphalis",
                "Inachis",
                "Araschnia",
                "Vanessa",
                "Euphydryas"
            ],
            "Heliconiinae": [
                "Argynnis",
                "Cethosia",
                "Childrena",
                "Speyeria",
                "Fabriciana",
                "Argyronome",
                "Issoria",
                "Brenthis",
                "Boloria",
                "Kuekenthaliella",
                "Clossiana",
                "Proclossiana"
            ],
            "Nemeobiinae": [
                "Hamearis",
                "Polycaena"
            ],
            "Theclinae": [
                "Favonius",
                "Tomares",
                "Chrysozephyrus",
                "Ussuriana",
                "Coreana",
                "Japonica",
                "Thecla",
                "Laeosopis",
                "Callophrys",
                "Satyrium",
                "Neolycaena"
            ],
            "Aphnaeinae": [
                "Cigaritis"
            ],
            "Polyommatinae": [
                "Celastrina",
                "Zizeeria",
                "Tarucus",
                "Cyclyrius",
                "Leptotes",
                "Lampides",
                "Cupido",
                "Maculinea",
                "Glaucopsyche",
                "Pseudophilotes",
                "Scolitantides",
                "Iolana",
                "Plebejus",
                "Agriades",
                "Plebejidea",
                "Kretania",
                "Aricia",
                "Pamiria",
                "Polyommatus",
                "Eumedonia",
                "Cyaniris",
                "Lysandra",
                "Glabroculus",
                "Neolysandra"
            ],
            "Libytheinae": [
                "Libythea"
            ],
            "Danainae": [
                "Danaus"
            ],
            "Charaxinae": [
                "Charaxes"
            ],
            "Apaturinae": [
                "Apatura",
                "Euapatura",
                "Hestina",
                "Timelaea",
                "Mimathyma"
            ],
            "Limenitidinae": [
                "Limenitis",
                "Lelecella",
                "Neptis"
            ]
        }

        self.child_of_genus = {
            "Carterocephalus": [
                "Carterocephalus_palaemon"
            ],
            "Heteropterus": [
                "Heteropterus_morpheus"
            ],
            "Thymelicus": [
                "Thymelicus_sylvestris",
                "Thymelicus_lineola",
                "Thymelicus_acteon"
            ],
            "Hesperia": [
                "Hesperia_comma"
            ],
            "Ochlodes": [
                "Ochlodes_venata"
            ],
            "Gegenes": [
                "Gegenes_nostrodamus"
            ],
            "Erynnis": [
                "Erynnis_tages"
            ],
            "Carcharodus": [
                "Carcharodus_alceae",
                "Carcharodus_lavatherae",
                "Carcharodus_baeticus",
                "Carcharodus_floccifera"
            ],
            "Spialia": [
                "Spialia_sertorius",
                "Spialia_orbifer"
            ],
            "Muschampia": [
                "Muschampia_proto"
            ],
            "Pyrgus": [
                "Pyrgus_alveus",
                "Pyrgus_armoricanus",
                "Pyrgus_andromedae",
                "Pyrgus_cacaliae",
                "Pyrgus_carlinae",
                "Pyrgus_carthami",
                "Pyrgus_malvae",
                "Pyrgus_cinarae",
                "Pyrgus_cirsii",
                "Pyrgus_malvoides",
                "Pyrgus_onopordi",
                "Pyrgus_serratulae",
                "Pyrgus_sidae",
                "Pyrgus_warrenensis"
            ],
            "Parnassius": [
                "Parnassius_sacerdos",
                "Parnassius_apollo",
                "Parnassius_mnemosyne",
                "Parnassius_glacialis",
                "Parnassius_stubbendorfii",
                "Parnassius_apollonius",
                "Parnassius_eversmannii",
                "Parnassius_ariadne",
                "Parnassius_stenosemus",
                "Parnassius_hardwickii",
                "Parnassius_charltonius",
                "Parnassius_imperator",
                "Parnassius_acdestis",
                "Parnassius_cardinal",
                "Parnassius_szechenyii",
                "Parnassius_delphius",
                "Parnassius_maximinus",
                "Parnassius_staudingeri",
                "Parnassius_orleans",
                "Parnassius_augustus",
                "Parnassius_loxias",
                "Parnassius_charltontonius",
                "Parnassius_autocrator",
                "Parnassius_stoliczkanus",
                "Parnassius_nordmanni",
                "Parnassius_simo",
                "Parnassius_bremeri",
                "Parnassius_actius",
                "Parnassius_cephalus",
                "Parnassius_maharaja",
                "Parnassius_tenedius",
                "Parnassius_acco",
                "Parnassius_boedromius",
                "Parnassius_tianschanicus",
                "Parnassius_phoebus",
                "Parnassius_honrathi",
                "Parnassius_ruckbeili",
                "Parnassius_epaphus",
                "Parnassius_nomion",
                "Parnassius_jacquemonti",
                "Parnassius_mercurius",
                "Parnassius_tibetanus",
                "Parnassius_clodius",
                "Parnassius_smintheus",
                "Parnassius_behrii"
            ],
            "Archon": [
                "Archon_apollinus"
            ],
            "Sericinus": [
                "Sericinus_montela"
            ],
            "Zerynthia": [
                "Zerynthia_rumina",
                "Zerynthia_polyxena"
            ],
            "Allancastria": [
                "Allancastria_cerisyi",
                "Allancastria_deyrollei",
                "Allancastria_caucasica"
            ],
            "Bhutanitis": [
                "Bhutanitis_thaidina",
                "Bhutanitis_lidderdalii",
                "Bhutanitis_mansfieldi"
            ],
            "Luehdorfia": [
                "Luehdorfia_japonica",
                "Luehdorfia_puziloi",
                "Luehdorfia_chinensis"
            ],
            "Papilio": [
                "Papilio_machaon",
                "Papilio_alexanor",
                "Papilio_hospiton",
                "Papilio_xuthus",
                "Papilio_dehaani",
                "Papilio_polytes",
                "Papilio_bootes",
                "Papilio_maackii",
                "Papilio_helenus",
                "Papilio_bianor",
                "Papilio_paris",
                "Papilio_krishna",
                "Papilio_macilentus",
                "Papilio_protenor",
                "Papilio_castor",
                "Papilio_arcturus"
            ],
            "Iphiclides": [
                "Iphiclides_podalirius",
                "Iphiclides_feisthamelii"
            ],
            "Leptidea": [
                "Leptidea_sinapis",
                "Leptidea_juvernica",
                "Leptidea_morsei",
                "Leptidea_amurensis",
                "Leptidea_duponcheli"
            ],
            "Colias": [
                "Colias_palaeno",
                "Colias_pelidne",
                "Colias_marcopolo",
                "Colias_ladakensis",
                "Colias_nebulosa",
                "Colias_nastes",
                "Colias_cocandica",
                "Colias_sieversi",
                "Colias_sifanica",
                "Colias_alpherakii",
                "Colias_christophi",
                "Colias_tyche",
                "Colias_phicomone",
                "Colias_alfacariensis",
                "Colias_hyale",
                "Colias_erate",
                "Colias_erschoffi",
                "Colias_romanovi",
                "Colias_regia",
                "Colias_stoliczkana",
                "Colias_hecla",
                "Colias_eogene",
                "Colias_thisoa",
                "Colias_staudingeri",
                "Colias_lada",
                "Colias_baeckeri",
                "Colias_fieldii",
                "Colias_heos",
                "Colias_caucasica",
                "Colias_diva",
                "Colias_chrysotheme",
                "Colias_balcanica",
                "Colias_myrmidone",
                "Colias_croceus",
                "Colias_felderi",
                "Colias_viluiensis",
                "Colias_aurorina",
                "Colias_chlorocoma",
                "Colias_libanotica",
                "Colias_wiskotti"
            ],
            "Aporia": [
                "Aporia_crataegi",
                "Aporia_procris",
                "Aporia_potanini",
                "Aporia_nabellica"
            ],
            "Catopsilia": [
                "Catopsilia_florella"
            ],
            "Gonepteryx": [
                "Gonepteryx_rhamni",
                "Gonepteryx_maxima",
                "Gonepteryx_cleopatra",
                "Gonepteryx_cleobule",
                "Gonepteryx_amintha",
                "Gonepteryx_farinosa"
            ],
            "Mesapia": [
                "Mesapia_peloria"
            ],
            "Baltia": [
                "Baltia_butleri"
            ],
            "Pieris": [
                "Pieris_brassicae",
                "Pieris_cheiranthi",
                "Pieris_rapae",
                "Pieris_mannii",
                "Pieris_ergane",
                "Pieris_krueperi",
                "Pieris_napi",
                "Pieris_bryoniae"
            ],
            "Erebia": [
                "Erebia_gorge",
                "Erebia_aethiopellus",
                "Erebia_mnestra",
                "Erebia_epistygne",
                "Erebia_ottomana",
                "Erebia_tyndarus",
                "Erebia_oeme",
                "Erebia_lefebvrei",
                "Erebia_melas",
                "Erebia_zapateri",
                "Erebia_neoridas",
                "Erebia_montana",
                "Erebia_cassioides",
                "Erebia_nivalis",
                "Erebia_scipio",
                "Erebia_pronoe",
                "Erebia_styx",
                "Erebia_meolans",
                "Erebia_palarica",
                "Erebia_pandrose",
                "Erebia_meta",
                "Erebia_erinnyn",
                "Erebia_euryale",
                "Erebia_ligea",
                "Erebia_eriphyle",
                "Erebia_manto",
                "Erebia_epiphron",
                "Erebia_flavofasciata",
                "Erebia_bubastis",
                "Erebia_claudina",
                "Erebia_christi",
                "Erebia_pharte",
                "Erebia_aethiops",
                "Erebia_melampus",
                "Erebia_sudetica",
                "Erebia_neriene",
                "Erebia_triaria",
                "Erebia_medusa",
                "Erebia_alberganus",
                "Erebia_pluto"
            ],
            "Berberia": [
                "Berberia_lambessanus",
                "Berberia_abdelkader"
            ],
            "Proterebia": [
                "Proterebia_afra"
            ],
            "Boeberia": [
                "Boeberia_parmenio"
            ],
            "Loxerebia": [
                "Loxerebia_saxicola"
            ],
            "Lycaena": [
                "Lycaena_thersamon",
                "Lycaena_lampon",
                "Lycaena_solskyi",
                "Lycaena_splendens",
                "Lycaena_candens",
                "Lycaena_ochimus",
                "Lycaena_hippothoe",
                "Lycaena_tityrus",
                "Lycaena_thetis",
                "Lycaena_phlaeas",
                "Lycaena_helle",
                "Lycaena_pang",
                "Lycaena_caspius",
                "Lycaena_margelanica",
                "Lycaena_dispar",
                "Lycaena_alciphron",
                "Lycaena_virgaureae",
                "Lycaena_kasyapa"
            ],
            "Melitaea": [
                "Melitaea_athalia",
                "Melitaea_parthenoides",
                "Melitaea_cinxia",
                "Melitaea_phoebe",
                "Melitaea_didyma",
                "Melitaea_varia",
                "Melitaea_aurelia",
                "Melitaea_asteria",
                "Melitaea_diamina",
                "Melitaea_britomartis",
                "Melitaea_acraeina",
                "Melitaea_trivia",
                "Melitaea_persea",
                "Melitaea_ambigua",
                "Melitaea_deione",
                "Melitaea_turanica",
                "Melitaea_nevadensis"
            ],
            "Argynnis": [
                "Argynnis_paphia",
                "Argynnis_pandora"
            ],
            "Heliophorus": [
                "Heliophorus_tamu",
                "Heliophorus_brahma",
                "Heliophorus_androcles"
            ],
            "Cethosia": [
                "Cethosia_biblis"
            ],
            "Childrena": [
                "Childrena_childreni"
            ],
            "Pontia": [
                "Pontia_edusa",
                "Pontia_daplidice",
                "Pontia_callidice",
                "Pontia_chloridice"
            ],
            "Anthocharis": [
                "Anthocharis_thibetana",
                "Anthocharis_bambusarum",
                "Anthocharis_bieti",
                "Anthocharis_scolymus",
                "Anthocharis_gruneri",
                "Anthocharis_damone",
                "Anthocharis_cardamines",
                "Anthocharis_belia",
                "Anthocharis_euphenoides"
            ],
            "Zegris": [
                "Zegris_pyrothoe",
                "Zegris_eupheme",
                "Zegris_fausti"
            ],
            "Euchloe": [
                "Euchloe_simplonia",
                "Euchloe_belemia",
                "Euchloe_ausonia",
                "Euchloe_tagis",
                "Euchloe_crameri",
                "Euchloe_insularis",
                "Euchloe_orientalis",
                "Euchloe_transcaspica",
                "Euchloe_charlonia",
                "Euchloe_tomyris"
            ],
            "Colotis": [
                "Colotis_fausta",
                "Colotis_evagore"
            ],
            "Hamearis": [
                "Hamearis_lucina"
            ],
            "Polycaena": [
                "Polycaena_tamerlana"
            ],
            "Favonius": [
                "Favonius_quercus"
            ],
            "Cigaritis": [
                "Cigaritis_siphax",
                "Cigaritis_allardi"
            ],
            "Tomares": [
                "Tomares_ballus",
                "Tomares_nogelii",
                "Tomares_mauretanicus",
                "Tomares_romanovi",
                "Tomares_callimachus"
            ],
            "Chrysozephyrus": [
                "Chrysozephyrus_smaragdinus"
            ],
            "Ussuriana": [
                "Ussuriana_micahaelis"
            ],
            "Coreana": [
                "Coreana_raphaelis"
            ],
            "Japonica": [
                "Japonica_saepestriata"
            ],
            "Thecla": [
                "Thecla_betulae"
            ],
            "Celastrina": [
                "Celastrina_argiolus"
            ],
            "Laeosopis": [
                "Laeosopis_roboris"
            ],
            "Callophrys": [
                "Callophrys_rubi",
                "Callophrys_avis"
            ],
            "Zizeeria": [
                "Zizeeria_knysna"
            ],
            "Tarucus": [
                "Tarucus_theophrastus",
                "Tarucus_balkanica"
            ],
            "Cyclyrius": [
                "Cyclyrius_webbianus"
            ],
            "Leptotes": [
                "Leptotes_pirithous"
            ],
            "Satyrium": [
                "Satyrium_spini",
                "Satyrium_w-album",
                "Satyrium_ilicis",
                "Satyrium_pruni",
                "Satyrium_acaciae",
                "Satyrium_esculi"
            ],
            "Lampides": [
                "Lampides_boeticus"
            ],
            "Neolycaena": [
                "Neolycaena_rhymnus"
            ],
            "Cupido": [
                "Cupido_minimus",
                "Cupido_alcetas",
                "Cupido_osiris",
                "Cupido_argiades",
                "Cupido_decolorata"
            ],
            "Maculinea": [
                "Maculinea_rebeli",
                "Maculinea_arion",
                "Maculinea_alcon",
                "Maculinea_teleius",
                "Maculinea_nausithous"
            ],
            "Glaucopsyche": [
                "Glaucopsyche_melanops",
                "Glaucopsyche_alexis",
                "Glaucopsyche_paphos"
            ],
            "Pseudophilotes": [
                "Pseudophilotes_abencerragus",
                "Pseudophilotes_panoptes",
                "Pseudophilotes_vicrama",
                "Pseudophilotes_baton"
            ],
            "Scolitantides": [
                "Scolitantides_orion"
            ],
            "Iolana": [
                "Iolana_gigantea",
                "Iolana_iolas"
            ],
            "Plebejus": [
                "Plebejus_argus",
                "Plebejus_eversmanni",
                "Plebejus_argyrognomon",
                "Plebejus_idas",
                "Plebejus_samudra"
            ],
            "Agriades": [
                "Agriades_optilete",
                "Agriades_orbitulus",
                "Agriades_glandon",
                "Agriades_pheretiades",
                "Agriades_lehanus"
            ],
            "Plebejidea": [
                "Plebejidea_loewii"
            ],
            "Kretania": [
                "Kretania_trappi",
                "Kretania_pylaon",
                "Kretania_martini"
            ],
            "Aricia": [
                "Aricia_artaxerxes",
                "Aricia_agestis",
                "Aricia_nicias",
                "Aricia_anteros"
            ],
            "Pamiria": [
                "Pamiria_omphisa"
            ],
            "Polyommatus": [
                "Polyommatus_damon",
                "Polyommatus_dolus",
                "Polyommatus_antidolus",
                "Polyommatus_phyllis",
                "Polyommatus_iphidamon",
                "Polyommatus_damonides",
                "Polyommatus_damone",
                "Polyommatus_ripartii",
                "Polyommatus_admetus",
                "Polyommatus_dorylas",
                "Polyommatus_erschoffi",
                "Polyommatus_thersites",
                "Polyommatus_escheri",
                "Polyommatus_nivescens",
                "Polyommatus_aedon",
                "Polyommatus_atys",
                "Polyommatus_icarus",
                "Polyommatus_caeruleus",
                "Polyommatus_stoliczkana",
                "Polyommatus_golgus",
                "Polyommatus_amandus",
                "Polyommatus_daphnis",
                "Polyommatus_eros",
                "Polyommatus_celina"
            ],
            "Eumedonia": [
                "Eumedonia_eumedon"
            ],
            "Cyaniris": [
                "Cyaniris_semiargus"
            ],
            "Lysandra": [
                "Lysandra_bellargus",
                "Lysandra_coridon",
                "Lysandra_hispana",
                "Lysandra_albicans",
                "Lysandra_caelestissima",
                "Lysandra_punctifera"
            ],
            "Glabroculus": [
                "Glabroculus_elvira",
                "Glabroculus_cyane"
            ],
            "Neolysandra": [
                "Neolysandra_coelestina",
                "Neolysandra_corona"
            ],
            "Libythea": [
                "Libythea_celtis"
            ],
            "Danaus": [
                "Danaus_plexippus",
                "Danaus_chrysippus"
            ],
            "Charaxes": [
                "Charaxes_jasius"
            ],
            "Apatura": [
                "Apatura_iris",
                "Apatura_ilia",
                "Apatura_metis"
            ],
            "Limenitis": [
                "Limenitis_reducta",
                "Limenitis_populi",
                "Limenitis_camilla",
                "Limenitis_sydyi"
            ],
            "Euapatura": [
                "Euapatura_mirza"
            ],
            "Hestina": [
                "Hestina_japonica"
            ],
            "Timelaea": [
                "Timelaea_albescens"
            ],
            "Mimathyma": [
                "Mimathyma_schrenckii"
            ],
            "Lelecella": [
                "Lelecella_limenitoides"
            ],
            "Neptis": [
                "Neptis_sappho",
                "Neptis_rivularis"
            ],
            "Nymphalis": [
                "Nymphalis_antiopa",
                "Nymphalis_polychloros",
                "Nymphalis_xanthomelas",
                "Nymphalis_l-album",
                "Nymphalis_urticae",
                "Nymphalis_ichnusa",
                "Nymphalis_egea",
                "Nymphalis_c-album",
                "Nymphalis_canace",
                "Nymphalis_c-aureum"
            ],
            "Inachis": [
                "Inachis_io"
            ],
            "Araschnia": [
                "Araschnia_burejana",
                "Araschnia_levana"
            ],
            "Vanessa": [
                "Vanessa_atalanta",
                "Vanessa_vulcania",
                "Vanessa_cardui"
            ],
            "Speyeria": [
                "Speyeria_aglaja",
                "Speyeria_clara"
            ],
            "Fabriciana": [
                "Fabriciana_niobe",
                "Fabriciana_adippe",
                "Fabriciana_jainadeva",
                "Fabriciana_auresiana",
                "Fabriciana_elisa"
            ],
            "Argyronome": [
                "Argyronome_laodice"
            ],
            "Issoria": [
                "Issoria_lathonia"
            ],
            "Brenthis": [
                "Brenthis_hecate",
                "Brenthis_daphne",
                "Brenthis_ino"
            ],
            "Boloria": [
                "Boloria_pales",
                "Boloria_aquilonaris",
                "Boloria_napaea",
                "Boloria_graeca"
            ],
            "Kuekenthaliella": [
                "Kuekenthaliella_eugenia"
            ],
            "Clossiana": [
                "Clossiana_selene",
                "Clossiana_thore",
                "Clossiana_dia",
                "Clossiana_euphrosyne",
                "Clossiana_titania",
                "Clossiana_freija"
            ],
            "Proclossiana": [
                "Proclossiana_eunomia"
            ],
            "Euphydryas": [
                "Euphydryas_maturna",
                "Euphydryas_ichnea",
                "Euphydryas_cynthia",
                "Euphydryas_aurinia",
                "Euphydryas_sibirica",
                "Euphydryas_iduna"
            ],
            "Melanargia": [
                "Melanargia_titea",
                "Melanargia_parce",
                "Melanargia_lachesis",
                "Melanargia_galathea",
                "Melanargia_russiae",
                "Melanargia_larissa",
                "Melanargia_ines",
                "Melanargia_pherusa",
                "Melanargia_occitanica",
                "Melanargia_arge",
                "Melanargia_meridionalis",
                "Melanargia_leda",
                "Melanargia_halimede"
            ],
            "Davidina": [
                "Davidina_armandi"
            ],
            "Hipparchia": [
                "Hipparchia_semele",
                "Hipparchia_parisatis",
                "Hipparchia_fidia",
                "Hipparchia_genava",
                "Hipparchia_aristaeus",
                "Hipparchia_fagi",
                "Hipparchia_wyssii",
                "Hipparchia_fatua",
                "Hipparchia_statilinus",
                "Hipparchia_syriaca",
                "Hipparchia_neomiris",
                "Hipparchia_azorina"
            ],
            "Chazara": [
                "Chazara_briseis",
                "Chazara_prieuri",
                "Chazara_bischoffii",
                "Chazara_persephone"
            ],
            "Pseudochazara": [
                "Pseudochazara_pelopea",
                "Pseudochazara_beroe",
                "Pseudochazara_schahrudensis",
                "Pseudochazara_telephassa",
                "Pseudochazara_anthelea",
                "Pseudochazara_amalthea",
                "Pseudochazara_graeca",
                "Pseudochazara_cingovskii"
            ],
            "Karanasa": [
                "Karanasa_modesta"
            ],
            "Oeneis": [
                "Oeneis_magna",
                "Oeneis_glacialis"
            ],
            "Satyrus": [
                "Satyrus_actaea",
                "Satyrus_parthicus",
                "Satyrus_ferula"
            ],
            "Minois": [
                "Minois_dryas"
            ],
            "Arethusana": [
                "Arethusana_arethusa"
            ],
            "Brintesia": [
                "Brintesia_circe"
            ],
            "Maniola": [
                "Maniola_jurtina",
                "Maniola_nurag"
            ],
            "Aphantopus": [
                "Aphantopus_hyperantus"
            ],
            "Hyponephele": [
                "Hyponephele_pulchra",
                "Hyponephele_pulchella",
                "Hyponephele_cadusia",
                "Hyponephele_amardaea",
                "Hyponephele_lycaon",
                "Hyponephele_lupina"
            ],
            "Pyronia": [
                "Pyronia_tithonus",
                "Pyronia_bathseba",
                "Pyronia_cecilia",
                "Pyronia_janiroides"
            ],
            "Coenonympha": [
                "Coenonympha_gardetta",
                "Coenonympha_tullia",
                "Coenonympha_corinna",
                "Coenonympha_pamphilus",
                "Coenonympha_dorus",
                "Coenonympha_darwiniana",
                "Coenonympha_arcania",
                "Coenonympha_leander",
                "Coenonympha_iphioides",
                "Coenonympha_glycerion",
                "Coenonympha_hero",
                "Coenonympha_oedippus"
            ],
            "Pararge": [
                "Pararge_aegeria",
                "Pararge_xiphioides"
            ],
            "Ypthima": [
                "Ypthima_baldus"
            ],
            "Lasiommata": [
                "Lasiommata_megera",
                "Lasiommata_petropolitana",
                "Lasiommata_maera",
                "Lasiommata_paramegaera"
            ],
            "Lopinga": [
                "Lopinga_achine"
            ],
            "Kirinia": [
                "Kirinia_roxelana",
                "Kirinia_climene",
                "Kirinia_eversmanni"
            ],
            "Neope": [
                "Neope_goschkevitschii"
            ],
            "Atrophaneura": [
                "Atrophaneura_mencius",
                "Atrophaneura_plutonius",
                "Atrophaneura_horishana",
                "Atrophaneura_impediens",
                "Atrophaneura_polyeuctes",
                "Atrophaneura_alcinous",
                "Atrophaneura_nevilli"
            ],
            "Agehana": [
                "Agehana_elwesi"
            ],
            "Arisbe": [
                "Arisbe_mandarinus",
                "Arisbe_parus",
                "Arisbe_alebion",
                "Arisbe_eurous",
                "Arisbe_doson",
                "Arisbe_tamerlanus",
                "Arisbe_leechi"
            ],
            "Teinopalpus": [
                "Teinopalpus_imperialis"
            ],
            "Graphium": [
                "Graphium_sarpedon",
                "Graphium_cloanthus"
            ],
            "Meandrusa": [
                "Meandrusa_sciron"
            ]
        }

        self.levels = [len(self.family), len(self.subfamily), len(self.genus), len(self.genus_specific_epithet)]

        self.level_names = ['family', 'subfamily', 'genus', 'genus_specific_epithet']

class CubDataset(Dataset):
    def __init__(self, root, list_path, transform):
        super(CubDataset, self).__init__()
        name_list = []
        order_label_list = []
        family_label_list = []
        genus_label_list = []
        class_label_list = []
        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename, class_label, genus_label, family_label, order_label  = l.strip().split(' ')
                name_list.append(imagename)
                order_label_list.append(int(order_label))
                family_label_list.append(int(family_label))
                genus_label_list.append(int(genus_label))
                class_label_list.append(int(class_label))
        self.image_filenames = [os.path.join(root, x) for x in name_list]
        self.transform = transform
        self.order_label_list = order_label_list
        self.family_label_list = family_label_list
        self.genus_label_list = genus_label_list
        self.class_label_list = class_label_list
    
    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        inputs = Image.open(self.image_filenames[index]).convert('RGB')
        if self.transform:
            inputs = self.transform(inputs) 
        order_label = self.order_label_list[index] - 1
        family_label = self.family_label_list[index] - 1
        genus_label = self.genus_label_list[index] - 1
        class_label = self.class_label_list[index] - 1

        return inputs, order_label, family_label, genus_label, class_label

    def __len__(self):
        return len(self.image_filenames)

class StanfordCars(VisionDataset):

    def __init__(self, root, split, transform=None):
        super().__init__(root, transform=transform)

        self.split = split
        self._base_folder = os.path.join(root, "stanford_cars")
        devkit = os.path.join(self._base_folder, "devkit")

        if self.split == "train":
            self._annotations_mat_path = os.path.join(devkit, "cars_train_annos.mat")
            self._images_base_path = os.path.join(self._base_folder, "cars_train")
        else:
            self._annotations_mat_path = os.path.join(self._base_folder, "cars_test_annos_withlabels.mat")
            self._images_base_path = os.path.join(self._base_folder, "cars_test")

        self._samples = [
            (
                os.path.join(self._images_base_path, annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(os.path.join(devkit, "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.trees = [
                [1, 7],
                [2, 6],
                [3, 6],
                [4, 6],
                [5, 6],
                [6, 3],
                [7, 4],
                [8, 2],
                [9, 3],
                [10, 2],
                [11, 3],
                [12, 2],
                [13, 3],
                [14, 3],
                [15, 3],
                [16, 6],
                [17, 6],
                [18, 9],
                [19, 4],
                [20, 6],
                [21, 2],
                [22, 3],
                [23, 6],
                [24, 6],
                [25, 3],
                [26, 6],
                [27, 2],
                [28, 3],
                [29, 6],
                [30, 9],
                [31, 2],
                [32, 7],
                [33, 7],
                [34, 3],
                [35, 6],
                [36, 2],
                [37, 7],
                [38, 2],
                [39, 2],
                [40, 6],
                [41, 6],
                [42, 3],
                [43, 3],
                [44, 6],
                [45, 2],
                [46, 3],
                [47, 6],
                [48, 7],
                [49, 6],
                [50, 7],
                [51, 6],
                [52, 7],
                [53, 1],
                [54, 1],
                [55, 2],
                [56, 3],
                [57, 3],
                [58, 7],
                [59, 2],
                [60, 5],
                [61, 6],
                [62, 7],
                [63, 6],
                [64, 8],
                [65, 1],
                [66, 3],
                [67, 6],
                [68, 7],
                [69, 1],
                [70, 1],
                [71, 8],
                [72, 3],
                [73, 6],
                [74, 1],
                [75, 1],
                [76, 7],
                [77, 2],
                [78, 5],
                [79, 6],
                [80, 2],
                [81, 2],
                [82, 9],
                [83, 9],
                [84, 9],
                [85, 5],
                [86, 1],
                [87, 1],
                [88, 8],
                [89, 7],
                [90, 1],
                [91, 1],
                [92, 9],
                [93, 3],
                [94, 7],
                [95, 7],
                [96, 6],
                [97, 6],
                [98, 4],
                [99, 3],
                [100, 2],
                [101, 3],
                [102, 2],
                [103, 2],
                [104, 3],
                [105, 6],
                [106, 1],
                [107, 2],
                [108, 5],
                [109, 7],
                [110, 7],
                [111, 1],
                [112, 3],
                [113, 1],
                [114, 1],
                [115, 6],
                [116, 9],
                [117, 6],
                [118, 7],
                [119, 8],
                [120, 7],
                [121, 7],
                [122, 1],
                [123, 2],
                [124, 1],
                [125, 1],
                [126, 5],
                [127, 5],
                [128, 3],
                [129, 6],
                [130, 4],
                [131, 7],
                [132, 7],
                [133, 7],
                [134, 6],
                [135, 6],
                [136, 6],
                [137, 6],
                [138, 6],
                [139, 4],
                [140, 6],
                [141, 3],
                [142, 7],
                [143, 7],
                [144, 3],
                [145, 7],
                [146, 7],
                [147, 7],
                [148, 7],
                [149, 7],
                [150, 3],
                [151, 3],
                [152, 3],
                [153, 3],
                [154, 7],
                [155, 7],
                [156, 6],
                [157, 2],
                [158, 2],
                [159, 7],
                [160, 3],
                [161, 2],
                [162, 6],
                [163, 3],
                [164, 6],
                [165, 6],
                [166, 8],
                [167, 6],
                [168, 4],
                [169, 8],
                [170, 4],
                [171, 3],
                [172, 3],
                [173, 6],
                [174, 5],
                [175, 2],
                [176, 6],
                [177, 6],
                [178, 4],
                [179, 2],
                [180, 3],
                [181, 6],
                [182, 6],
                [183, 4],
                [184, 6],
                [185, 6],
                [186, 7],
                [187, 6],
                [188, 6],
                [189, 7],
                [190, 4],
                [191, 4],
                [192, 4],
                [193, 4],
                [194, 6],
                [195, 7],
                [196, 2]
                ]
        self.coarse_classes = ["Cab", "Convertible", "Coupe", "Hatchback", "Minivan", "Sedan", "SUV", "Van", "Wagon"]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")
        global target_coarse

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        for i in range(len(self.trees)):
            if target + 1 == self.trees[i][0]:
                target_coarse = self.trees[i][1] - 1
        return pil_image, target, target_coarse

class StanfordCarsLabelMap:
    def __init__(self, root):
        self.trees = [
                [1, 7],
                [2, 6],
                [3, 6],
                [4, 6],
                [5, 6],
                [6, 3],
                [7, 4],
                [8, 2],
                [9, 3],
                [10, 2],
                [11, 3],
                [12, 2],
                [13, 3],
                [14, 3],
                [15, 3],
                [16, 6],
                [17, 6],
                [18, 9],
                [19, 4],
                [20, 6],
                [21, 2],
                [22, 3],
                [23, 6],
                [24, 6],
                [25, 3],
                [26, 6],
                [27, 2],
                [28, 3],
                [29, 6],
                [30, 9],
                [31, 2],
                [32, 7],
                [33, 7],
                [34, 3],
                [35, 6],
                [36, 2],
                [37, 7],
                [38, 2],
                [39, 2],
                [40, 6],
                [41, 6],
                [42, 3],
                [43, 3],
                [44, 6],
                [45, 2],
                [46, 3],
                [47, 6],
                [48, 7],
                [49, 6],
                [50, 7],
                [51, 6],
                [52, 7],
                [53, 1],
                [54, 1],
                [55, 2],
                [56, 3],
                [57, 3],
                [58, 7],
                [59, 2],
                [60, 5],
                [61, 6],
                [62, 7],
                [63, 6],
                [64, 8],
                [65, 1],
                [66, 3],
                [67, 6],
                [68, 7],
                [69, 1],
                [70, 1],
                [71, 8],
                [72, 3],
                [73, 6],
                [74, 1],
                [75, 1],
                [76, 7],
                [77, 2],
                [78, 5],
                [79, 6],
                [80, 2],
                [81, 2],
                [82, 9],
                [83, 9],
                [84, 9],
                [85, 5],
                [86, 1],
                [87, 1],
                [88, 8],
                [89, 7],
                [90, 1],
                [91, 1],
                [92, 9],
                [93, 3],
                [94, 7],
                [95, 7],
                [96, 6],
                [97, 6],
                [98, 4],
                [99, 3],
                [100, 2],
                [101, 3],
                [102, 2],
                [103, 2],
                [104, 3],
                [105, 6],
                [106, 1],
                [107, 2],
                [108, 5],
                [109, 7],
                [110, 7],
                [111, 1],
                [112, 3],
                [113, 1],
                [114, 1],
                [115, 6],
                [116, 9],
                [117, 6],
                [118, 7],
                [119, 8],
                [120, 7],
                [121, 7],
                [122, 1],
                [123, 2],
                [124, 1],
                [125, 1],
                [126, 5],
                [127, 5],
                [128, 3],
                [129, 6],
                [130, 4],
                [131, 7],
                [132, 7],
                [133, 7],
                [134, 6],
                [135, 6],
                [136, 6],
                [137, 6],
                [138, 6],
                [139, 4],
                [140, 6],
                [141, 3],
                [142, 7],
                [143, 7],
                [144, 3],
                [145, 7],
                [146, 7],
                [147, 7],
                [148, 7],
                [149, 7],
                [150, 3],
                [151, 3],
                [152, 3],
                [153, 3],
                [154, 7],
                [155, 7],
                [156, 6],
                [157, 2],
                [158, 2],
                [159, 7],
                [160, 3],
                [161, 2],
                [162, 6],
                [163, 3],
                [164, 6],
                [165, 6],
                [166, 8],
                [167, 6],
                [168, 4],
                [169, 8],
                [170, 4],
                [171, 3],
                [172, 3],
                [173, 6],
                [174, 5],
                [175, 2],
                [176, 6],
                [177, 6],
                [178, 4],
                [179, 2],
                [180, 3],
                [181, 6],
                [182, 6],
                [183, 4],
                [184, 6],
                [185, 6],
                [186, 7],
                [187, 6],
                [188, 6],
                [189, 7],
                [190, 4],
                [191, 4],
                [192, 4],
                [193, 4],
                [194, 6],
                [195, 7],
                [196, 2]
                ]
        self.coarse_classes = ["Cab", "Convertible", "Coupe", "Hatchback", "Minivan", "Sedan", "SUV", "Van", "Wagon"]
        self._base_folder = os.path.join(root, "stanford_cars")
        devkit = os.path.join(self._base_folder, "devkit")
        self.fine_classes = sio.loadmat(os.path.join(devkit, "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()

class AircraftDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_paths, self.variant_labels, self.family_labels, self.maker_labels = self.load_data()

    def load_data(self):
        image_paths = []
        variant_labels = []
        family_labels = []
        maker_labels = []

        with open(os.path.join(self.data_dir, f'images_variant_{self.split}.txt'), 'r') as variant_file, \
             open(os.path.join(self.data_dir, f'images_family_{self.split}.txt'), 'r') as family_file, \
             open(os.path.join(self.data_dir, f'images_manufacturer_{self.split}.txt'), 'r') as maker_file:
            for variant_line, family_line, maker_line in zip(variant_file, family_file, maker_file):
                image_name, variant_label = variant_line.strip().split(maxsplit=1)
                _, family_label = family_line.strip().split(maxsplit=1)
                _, maker_label = maker_line.strip().split(maxsplit=1)
                image_name = image_name + '.jpg'
                image_path = os.path.join(self.data_dir, 'images', image_name)

                image_paths.append(image_path)
                variant_labels.append(variant_label)
                family_labels.append(family_label)
                maker_labels.append(maker_label)

        return image_paths, variant_labels, family_labels, maker_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        variant_label = self.variant_labels[idx]
        family_label = self.family_labels[idx]
        maker_label = self.maker_labels[idx]

        return image, variant_label, family_label, maker_label

class AircraftMap:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels_names_variant = []
        self.labels_names_family = []
        self.labels_names_maker = []
        with open(os.path.join(self.data_dir, 'variants.txt'), 'r') as variant_file, \
             open(os.path.join(self.data_dir, 'families.txt'), 'r') as family_file, \
             open(os.path.join(self.data_dir, 'manufacturers.txt'), 'r') as maker_file:
            for variant_line in variant_file:
                variant_label = variant_line.strip()
                self.labels_names_variant.append(variant_label)

            for family_line in family_file:
                family_label = family_line.strip()
                self.labels_names_family.append(family_label)

            for maker_line in maker_file:
                maker_label = maker_line.strip()
                self.labels_names_maker.append(maker_label)

        self.trees = [
                        [1, 1, 1],
                        [2, 2, 1],
                        [3, 3, 1],
                        [4, 3, 1],
                        [5, 3, 1],
                        [6, 3, 1],
                        [7, 4, 1],
                        [8, 4, 1],
                        [9, 5, 1],
                        [10, 5, 1],
                        [11, 5, 1],
                        [12, 5, 1],
                        [13, 6, 1],
                        [14, 7, 2],
                        [15, 8, 3],
                        [16, 9, 3],
                        [17, 10, 7],
                        [18, 10, 7],
                        [19, 11, 7],
                        [20, 12, 4],
                        [21, 13, 5],
                        [22, 14, 5],
                        [23, 15, 5],
                        [24, 16, 5],
                        [25, 16, 5],
                        [26, 16, 5],
                        [27, 16, 5],
                        [28, 16, 5],
                        [29, 16, 5],
                        [30, 16, 5],
                        [31, 16, 5],
                        [32, 17, 5],
                        [33, 17, 5],
                        [34, 17, 5],
                        [35, 17, 5],
                        [36, 18, 5],
                        [37, 18, 5],
                        [38, 19, 5],
                        [39, 19, 5],
                        [40, 19, 5],
                        [41, 20, 5],
                        [42, 20, 5],
                        [43, 21, 21],
                        [44, 22, 14],
                        [45, 23, 9],
                        [46, 24, 9],
                        [47, 25, 9],
                        [48, 25, 9],
                        [49, 26, 8],
                        [50, 27, 8],
                        [51, 28, 8],
                        [52, 28, 8],
                        [53, 29, 12],
                        [54, 29, 12],
                        [55, 30, 23],
                        [56, 31, 14],
                        [57, 32, 14],
                        [58, 33, 14],
                        [59, 34, 23],
                        [60, 35, 12],
                        [61, 36, 12],
                        [62, 37, 12],
                        [63, 38, 13],
                        [64, 39, 26],
                        [65, 40, 15],
                        [66, 41, 15],
                        [67, 41, 15],
                        [68, 41, 15],
                        [69, 42, 15],
                        [70, 42, 15],
                        [71, 43, 15],
                        [72, 44, 16],
                        [73, 45, 23],
                        [74, 46, 22],
                        [75, 47, 11],
                        [76, 48, 11],
                        [77, 49, 18],
                        [78, 50, 18],
                        [79, 51, 18],
                        [80, 52, 6],
                        [81, 53, 19],
                        [82, 53, 19],
                        [83, 54, 7],
                        [84, 55, 20],
                        [85, 56, 4],
                        [86, 57, 21],
                        [87, 58, 23],
                        [88, 59, 23],
                        [89, 59, 23],
                        [90, 60, 23],
                        [91, 61, 17],
                        [92, 62, 25],
                        [93, 63, 27],
                        [94, 64, 27],
                        [95, 65, 28],
                        [96, 66, 10],
                        [97, 67, 24],
                        [98, 68, 29],
                        [99, 69, 29],
                        [100, 70, 30]
                    ]
    
class caltech(Dataset):
    def __init__(self, dataset, transform, hierarchy_file, labels_names_l3):
        self.dataset = dataset
        self.transform = transform
        self.hierarchy_file = hierarchy_file
        self.labels_names_l3 = labels_names_l3
        self.labels_names_l2, self.labels_names_l1 = self.load_data()
    
    def load_data(self):
        l2 = []
        l1 = []
        with open(self.hierarchy_file, 'r') as f:
            for i in f.readlines():
                level3, level2, level1 = i.split(',')
                l2.append(level2.lower().strip().replace('_',' ').replace('-',' '))
                l1.append(level1.lower().strip().replace('_',' ').replace('-',' '))
        return list(set(l2)), list(set(l1))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        def search_l2_l1(l3):
            l3 = l3.lower().strip().replace('_',' ').replace('-',' ')
            with open(self.hierarchy_file, 'r') as f:
                for i in f.readlines():
                    i = i.strip()
                    level3, level2, level1 = i.split(',')
                    if '256' in self.hierarchy_file:
                        level3 = level3.lower().strip().replace('-101','').replace('-',' ').replace('_',' ')
                    else:
                        level3 = level3.lower().strip().replace('_',' ')
                    level2 = level2.lower().strip().replace('-',' ').replace('_',' ')
                    level1 = level1.lower().strip().replace('-',' ').replace('_',' ')
                    if level3 == l3:
                        return level2, level1
        label_l3 = self.labels_names_l3[label]
        try:
            label_l2, label_l1 = search_l2_l1(label_l3)
        except:
            print(f'cannot find {label_l3}')
        return image, label_l3, label_l2, label_l1

class food101_dataset(Dataset):
    def __init__(self, dataset, transform, hierarchy_file, labels_names_l2):
        self.dataset = dataset
        self.transform = transform
        self.hierarchy_file = hierarchy_file
        self.labels_names_l2 = labels_names_l2
        self.labels_names_l1 = self.load_data()
    
    def load_data(self):
        l1 = []
        with open(self.hierarchy_file, 'r') as f:
            for i in f.readlines():
                level2, level1 = i.split(',')
                l1.append(level1.lower().strip().replace('_',' ').replace('-',' '))
        return list(set(l1))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        def search_l1(l2):
            l2 = l2.lower().strip().replace('_',' ').replace('-',' ')
            with open(self.hierarchy_file, 'r') as f:
                for i in f.readlines():
                    i = i.strip()
                    level2, level1 = i.split(',')
                    level2 = level2.lower().strip().replace('_',' ').replace('_',' ')
                    level1 = level1.lower().strip().replace('-',' ').replace('_',' ')
                    if level2 == l2:
                        return level1
        label_l2 = self.labels_names_l2[label]
        try:
            label_l1 = search_l1(label_l2)
        except:
            print(f'cannot find {label_l2}')
        return image, label_l2, label_l1

class BREEDSFactory:
    def __init__(self, args, info_dir, data_dir):
        self.args = args
        self.info_dir = info_dir
        self.data_dir = data_dir

    def get_breeds(self, split=None):
        superclasses, subclass_split, label_map = self.get_classes(self.args.dataset, split)
        self.label_map = label_map
        print(f"==> Preparing dataset {self.args.dataset}, split: {split}..")
        if split is not None:
            # split can be 'good','bad' or None. rif not None, 'subclass_split' will have 2 items, fo 'train' and 'test'. otherwise, just 1
            train_subclasses, test_subclasses = subclass_split
            # source 
            dataset_source = datasets.CustomImageNet(self.data_dir, train_subclasses)
            loaders_source = dataset_source.make_loaders(workers=2, batch_size=self.args.batch_size)
            train_loader_source, test_loader_source = loaders_source
            # target
            dataset_target = datasets.CustomImageNet(self.data_dir, test_subclasses)
            loaders_target = dataset_target.make_loaders(workers=2, batch_size=self.args.batch_size)
            _, test_loader_target = loaders_target
        else:
            raise NotImplementedError

        return train_loader_source, test_loader_source, test_loader_target

    def get_classes(self, ds_name, split=None):
        if ds_name == 'living17':
            return make_living17(self.info_dir, split)
        elif ds_name == 'entity30':
            return make_entity30(self.info_dir, split)
        elif ds_name == 'entity13':
            return make_entity13(self.info_dir, split)
        elif ds_name == 'nonliving26':
            return make_nonliving26(self.info_dir, split)
        else:
            raise NotImplementedError