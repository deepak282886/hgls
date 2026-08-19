"""
Synthetic Science Corpus Builder
Produces sentences.txt and pairs.jsonl from hand-authored
Simple English Wikipedia-style science articles.

Designed to give the graph training system:
- Dense causal chains (because → causes → leads to → results in)
- Shared vocabulary across domains (energy, heat, water, light)
- Repetition of core concepts so merges form quickly
- Cross-article concept links (energy appears in weather AND physics AND biology)
"""

import json
import re
from pathlib import Path

# ─────────────────────────────────────────────
# CORPUS — mini-articles, sentence by sentence
# Each list is one article. Order is preserved.
# ─────────────────────────────────────────────

ARTICLES = {

    # ── WEATHER & ATMOSPHERE ──────────────────────────────────────────────

    "The Water Cycle": [
        "the water cycle is the continuous movement of water through the environment.",
        "energy from the sun heats water on the surface of the earth.",
        "when water is heated, it evaporates and becomes water vapor.",
        "water vapor rises into the atmosphere because it is lighter than air.",
        "as water vapor rises, it cools and condenses into tiny water droplets.",
        "these droplets collect together to form clouds.",
        "when enough water droplets collect, they become heavy and fall as rain.",
        "rain flows into rivers, lakes, and oceans.",
        "some water soaks into the ground and becomes groundwater.",
        "the sun then heats the water again and the cycle continues.",
        "the water cycle moves water from the surface to the atmosphere and back.",
        "without the water cycle, life on earth could not survive.",
    ],

    "Clouds and Rain": [
        "clouds are made of millions of tiny water droplets or ice crystals.",
        "water droplets form when water vapor cools and condenses around dust particles.",
        "warm air rises and carries water vapor upward into cooler parts of the atmosphere.",
        "as the air cools, the water vapor condenses and forms clouds.",
        "different types of clouds form at different heights in the atmosphere.",
        "cumulus clouds are puffy white clouds that form at low altitudes.",
        "cumulonimbus clouds are tall storm clouds that produce heavy rain and lightning.",
        "rain forms when water droplets inside clouds combine and become too heavy to stay in the air.",
        "snow forms when temperatures inside clouds are below freezing.",
        "clouds reflect sunlight back into space and help keep the earth cool.",
        "the amount of cloud cover affects the temperature of the earth.",
    ],

    "The Atmosphere": [
        "the atmosphere is the layer of gases that surrounds the earth.",
        "the atmosphere is held in place by the force of gravity.",
        "the atmosphere is made mostly of nitrogen and oxygen.",
        "oxygen in the atmosphere allows animals and plants to breathe.",
        "the atmosphere protects the earth from harmful radiation from the sun.",
        "the ozone layer in the upper atmosphere absorbs ultraviolet radiation.",
        "without the ozone layer, ultraviolet radiation would damage living things.",
        "the atmosphere also traps heat from the sun and warms the earth.",
        "this warming effect is called the greenhouse effect.",
        "carbon dioxide and water vapor are the main gases that trap heat.",
        "when more carbon dioxide enters the atmosphere, more heat is trapped.",
        "this causes the temperature of the earth to rise.",
    ],

    "Wind and Air Pressure": [
        "wind is the movement of air from one place to another.",
        "air moves because of differences in air pressure.",
        "warm air rises because it is less dense than cold air.",
        "when warm air rises, cold air moves in to take its place.",
        "this movement of air creates wind.",
        "areas of high pressure have cold, sinking air.",
        "areas of low pressure have warm, rising air.",
        "air always moves from areas of high pressure to areas of low pressure.",
        "the greater the difference in pressure, the stronger the wind.",
        "wind carries heat and moisture from one part of the earth to another.",
        "wind affects the weather and climate of different regions.",
        "ocean currents are driven partly by wind blowing across the surface of the water.",
    ],

    "Temperature and Heat": [
        "temperature is a measure of how hot or cold something is.",
        "heat is a form of energy that flows from warmer objects to cooler objects.",
        "when an object absorbs heat energy, its temperature rises.",
        "when an object loses heat energy, its temperature falls.",
        "the sun is the main source of heat energy on earth.",
        "sunlight travels through space and warms the surface of the earth.",
        "dark surfaces absorb more heat than light surfaces.",
        "water absorbs and releases heat more slowly than land.",
        "this is why coastal areas have milder temperatures than inland areas.",
        "heat can travel by conduction, convection, and radiation.",
        "conduction transfers heat through direct contact between objects.",
        "convection transfers heat through the movement of liquids or gases.",
        "radiation transfers heat through electromagnetic waves without needing matter.",
    ],

    "Seasons and the Earth": [
        "the earth orbits the sun once every year.",
        "the earth is tilted on its axis at an angle of about 23 degrees.",
        "this tilt causes different parts of the earth to receive different amounts of sunlight.",
        "when the northern hemisphere is tilted toward the sun, it experiences summer.",
        "at the same time, the southern hemisphere is tilted away and experiences winter.",
        "in summer, days are longer and the sun is higher in the sky.",
        "more sunlight means more heat energy reaches the surface.",
        "in winter, days are shorter and the sun is lower in the sky.",
        "less sunlight means less heat energy reaches the surface, so temperatures are lower.",
        "near the equator, the sun is always high in the sky, so temperatures stay warm all year.",
        "the poles receive sunlight at a low angle, so they receive less heat and stay cold.",
    ],

    # ── PHYSICS ───────────────────────────────────────────────────────────

    "Forces and Motion": [
        "a force is a push or pull that acts on an object.",
        "forces can change the speed or direction of a moving object.",
        "when no force acts on an object, it stays still or continues moving in a straight line.",
        "this is called inertia.",
        "friction is a force that slows objects down when they slide against each other.",
        "without friction, objects would keep sliding forever.",
        "gravity is a force that pulls objects toward each other.",
        "the greater the mass of an object, the stronger its gravitational pull.",
        "when you drop an object, gravity pulls it toward the ground.",
        "newton's second law states that force equals mass times acceleration.",
        "a larger force causes a greater acceleration.",
        "a heavier object requires more force to accelerate at the same rate.",
    ],

    "Gravity": [
        "gravity is a force that attracts all objects with mass toward each other.",
        "the earth's gravity pulls objects toward the center of the earth.",
        "gravity is what keeps the moon orbiting the earth.",
        "gravity is what keeps the earth orbiting the sun.",
        "the strength of gravity depends on the mass of the objects and the distance between them.",
        "larger objects have stronger gravity.",
        "objects that are closer together feel a stronger gravitational pull.",
        "on the surface of the earth, gravity causes objects to fall with an acceleration of 9.8 meters per second squared.",
        "the moon has less mass than the earth, so its gravity is weaker.",
        "on the moon, objects fall more slowly than on earth.",
        "gravity holds the atmosphere close to the earth.",
        "without gravity, the atmosphere would drift away into space.",
    ],

    "Energy": [
        "energy is the ability to do work or cause change.",
        "energy exists in many different forms.",
        "kinetic energy is the energy of moving objects.",
        "potential energy is stored energy that can be released later.",
        "when an object falls, its potential energy is converted into kinetic energy.",
        "heat is a form of energy that transfers between objects at different temperatures.",
        "light is a form of energy that can travel through space.",
        "electrical energy is produced by the movement of electrons.",
        "chemical energy is stored in the bonds between atoms in molecules.",
        "energy cannot be created or destroyed, only converted from one form to another.",
        "this is called the law of conservation of energy.",
        "when fuel burns, chemical energy is converted into heat and light.",
        "plants convert light energy from the sun into chemical energy through photosynthesis.",
    ],

    "Light and Waves": [
        "light is a form of energy that travels in waves.",
        "light travels very fast — about 300 million meters per second.",
        "light can travel through empty space, unlike sound.",
        "when light hits an object, it can be absorbed, reflected, or transmitted.",
        "objects appear colored because they reflect certain wavelengths of light.",
        "when all wavelengths are reflected, an object appears white.",
        "when all wavelengths are absorbed, an object appears black.",
        "a prism separates white light into its component colors.",
        "the colors of visible light are red, orange, yellow, green, blue, and violet.",
        "different colors have different wavelengths and different amounts of energy.",
        "violet light has the shortest wavelength and the most energy.",
        "red light has the longest wavelength and the least energy.",
        "ultraviolet light has more energy than visible light and can damage living cells.",
    ],

    "Matter and Atoms": [
        "matter is anything that has mass and takes up space.",
        "all matter is made of tiny particles called atoms.",
        "atoms are the smallest units of an element that keep its chemical properties.",
        "atoms are made of a nucleus surrounded by electrons.",
        "the nucleus contains protons and neutrons.",
        "protons have a positive electric charge.",
        "electrons have a negative electric charge.",
        "neutrons have no electric charge.",
        "atoms of different elements have different numbers of protons.",
        "when atoms join together, they form molecules.",
        "water is a molecule made of two hydrogen atoms and one oxygen atom.",
        "the properties of a substance depend on how its atoms are arranged and bonded.",
        "matter can exist as a solid, liquid, or gas depending on temperature and pressure.",
    ],

    "States of Matter": [
        "matter can exist in three common states: solid, liquid, and gas.",
        "in a solid, particles are packed tightly together and vibrate in place.",
        "solids have a fixed shape and a fixed volume.",
        "in a liquid, particles are close together but can move past each other.",
        "liquids have a fixed volume but no fixed shape.",
        "in a gas, particles move freely and are spread far apart.",
        "gases have no fixed shape and no fixed volume.",
        "adding heat energy causes a solid to melt and become a liquid.",
        "adding more heat causes a liquid to evaporate and become a gas.",
        "removing heat causes a gas to condense into a liquid.",
        "removing more heat causes a liquid to freeze into a solid.",
        "the temperature at which a substance melts is called its melting point.",
        "the temperature at which a substance boils is called its boiling point.",
    ],

    "Electricity": [
        "electricity is produced by the movement of electrons.",
        "electrons are tiny particles that carry a negative electric charge.",
        "when electrons move through a material, they create an electric current.",
        "materials that allow electrons to flow through them are called conductors.",
        "metals are good conductors of electricity.",
        "materials that do not allow electrons to flow are called insulators.",
        "rubber and plastic are good insulators.",
        "a battery stores chemical energy and converts it into electrical energy.",
        "when a battery is connected in a circuit, electrons flow from one end to the other.",
        "this flow of electrons can power devices like lights and motors.",
        "the amount of energy carried by an electric current depends on the voltage.",
        "higher voltage drives more electrons through the circuit.",
    ],

    # ── BIOLOGY ───────────────────────────────────────────────────────────

    "Cells": [
        "all living things are made of cells.",
        "cells are the smallest units of life.",
        "each cell carries out the basic functions needed for life.",
        "cells take in nutrients and convert them into energy.",
        "cells grow, divide, and respond to their environment.",
        "the cell membrane surrounds the cell and controls what enters and leaves.",
        "the nucleus contains the genetic information of the cell.",
        "genetic information tells the cell how to grow and function.",
        "the mitochondria are the parts of the cell that produce energy.",
        "mitochondria convert nutrients into a form of energy the cell can use.",
        "plant cells have a cell wall outside the cell membrane that gives them shape.",
        "plant cells also contain chloroplasts, which capture light energy.",
    ],

    "Photosynthesis": [
        "photosynthesis is the process by which plants make their own food.",
        "plants use energy from sunlight to convert water and carbon dioxide into sugar.",
        "the sugar provides the plant with energy for growth and other functions.",
        "photosynthesis takes place in the chloroplasts inside plant cells.",
        "chloroplasts contain a green pigment called chlorophyll.",
        "chlorophyll absorbs light energy from the sun.",
        "water is absorbed by the roots of the plant and travels up to the leaves.",
        "carbon dioxide enters the leaves through tiny holes called stomata.",
        "inside the chloroplast, light energy is used to split water molecules.",
        "the hydrogen from water combines with carbon dioxide to form sugar.",
        "oxygen is released as a byproduct of photosynthesis.",
        "the oxygen released by plants is what animals breathe.",
        "without photosynthesis, there would be no oxygen in the atmosphere.",
    ],

    "Respiration": [
        "respiration is the process by which living things release energy from food.",
        "during respiration, glucose is broken down to release energy.",
        "the energy released is used to power all the activities of the cell.",
        "in aerobic respiration, glucose reacts with oxygen to produce energy.",
        "carbon dioxide and water are produced as waste products.",
        "the carbon dioxide is released and the water is used or expelled.",
        "aerobic respiration is more efficient than anaerobic respiration.",
        "anaerobic respiration occurs when oxygen is not available.",
        "during anaerobic respiration, less energy is released from glucose.",
        "lactic acid is produced as a waste product in animal cells during anaerobic respiration.",
        "the lactic acid that builds up in muscles causes the burning sensation during intense exercise.",
        "plants also carry out respiration to release the energy stored during photosynthesis.",
    ],

    "DNA and Heredity": [
        "dna is the molecule that carries the genetic information of living things.",
        "dna is found in the nucleus of every cell.",
        "dna is made of a long chain of smaller molecules called nucleotides.",
        "the sequence of nucleotides in dna carries the instructions for building proteins.",
        "proteins carry out most of the functions in living cells.",
        "genes are sections of dna that code for specific proteins.",
        "humans have about 20,000 genes in their dna.",
        "when a cell divides, it copies its dna so each new cell gets a complete set.",
        "organisms inherit their dna from their parents.",
        "half of an organism's dna comes from its mother and half from its father.",
        "variations in dna lead to differences between individuals.",
        "natural selection acts on these variations over many generations.",
        "over time, natural selection can lead to the evolution of new species.",
    ],

    "Evolution and Natural Selection": [
        "evolution is the process by which species change over time.",
        "charles darwin proposed the theory of evolution by natural selection.",
        "natural selection occurs because individuals in a population vary in their traits.",
        "some traits help individuals survive and reproduce more successfully than others.",
        "individuals with advantageous traits are more likely to survive and pass on their traits.",
        "over many generations, advantageous traits become more common in the population.",
        "disadvantageous traits become less common over time.",
        "over very long periods of time, populations can change so much that new species form.",
        "all living species on earth share common ancestors.",
        "the evidence for evolution comes from fossils, dna, and the anatomy of living organisms.",
        "fossils show how ancient species were different from modern species.",
        "similar dna sequences in different species show that they are related.",
    ],

    "Ecosystems": [
        "an ecosystem is a community of living things interacting with their environment.",
        "ecosystems include both living organisms and nonliving factors such as water, soil, and sunlight.",
        "producers are organisms that make their own food using energy from sunlight.",
        "plants and algae are the main producers in most ecosystems.",
        "consumers are organisms that get energy by eating other organisms.",
        "herbivores eat only plants.",
        "carnivores eat only animals.",
        "omnivores eat both plants and animals.",
        "decomposers break down dead organisms and return nutrients to the soil.",
        "energy flows through an ecosystem from producers to consumers to decomposers.",
        "each step in the chain loses about 90 percent of the energy as heat.",
        "this is why there are fewer large carnivores than small herbivores in any ecosystem.",
        "if one species is removed from an ecosystem, it can affect all the other species.",
    ],

    "The Human Body": [
        "the human body is made of trillions of cells organized into tissues and organs.",
        "the digestive system breaks down food into nutrients the body can use.",
        "the circulatory system carries nutrients and oxygen to all parts of the body.",
        "the heart pumps blood through a network of blood vessels.",
        "the lungs take in oxygen from the air and release carbon dioxide.",
        "the nervous system carries signals between the brain and the rest of the body.",
        "the brain processes information and controls the actions of the body.",
        "the skeletal system gives the body structure and protects internal organs.",
        "muscles are attached to bones and allow the body to move.",
        "the immune system protects the body from harmful bacteria and viruses.",
        "when the body detects an infection, white blood cells attack and destroy the invaders.",
        "the endocrine system releases hormones that regulate growth and body functions.",
    ],

    # ── EARTH & GEOLOGY ───────────────────────────────────────────────────

    "The Earth": [
        "the earth is a rocky planet that orbits the sun.",
        "the earth is the only planet known to support life.",
        "the earth is made of several layers: the crust, the mantle, and the core.",
        "the crust is the thin outer layer where we live.",
        "beneath the crust is the mantle, which is made of hot, solid rock.",
        "the core at the center of the earth is made mostly of iron and nickel.",
        "the outer core is liquid and the inner core is solid.",
        "the movement of liquid iron in the outer core creates earth's magnetic field.",
        "the magnetic field protects the earth from harmful particles from the sun.",
        "heat from the earth's core drives the movement of tectonic plates.",
        "tectonic plates are large sections of the crust that move slowly over the mantle.",
        "the movement of tectonic plates causes earthquakes and volcanic eruptions.",
    ],

    "Volcanoes": [
        "a volcano is an opening in the earth's crust through which molten rock can escape.",
        "molten rock beneath the earth's surface is called magma.",
        "when magma reaches the surface, it is called lava.",
        "magma collects in chambers beneath the surface until pressure builds up.",
        "when the pressure becomes great enough, the magma is forced up through the opening.",
        "volcanic eruptions can release large amounts of ash, gas, and lava.",
        "ash from volcanoes can block sunlight and lower temperatures on earth.",
        "over time, cooled lava builds up to form volcanic mountains.",
        "many islands, including hawaii, were formed by underwater volcanic eruptions.",
        "volcanic soils are very fertile because they are rich in minerals.",
        "volcanoes also release gases including water vapor and carbon dioxide into the atmosphere.",
    ],

    "Oceans": [
        "the oceans cover about 71 percent of the earth's surface.",
        "the oceans play a major role in regulating the earth's climate.",
        "water has a high capacity to absorb and store heat.",
        "the oceans absorb much of the heat from the sun and release it slowly.",
        "this moderates temperatures around the world.",
        "ocean currents carry warm water from the tropics toward the poles.",
        "this transfers heat energy from warm regions to cold regions.",
        "the oceans absorb carbon dioxide from the atmosphere.",
        "this helps reduce the greenhouse effect.",
        "phytoplankton in the ocean produce about half of the world's oxygen.",
        "phytoplankton carry out photosynthesis using sunlight and carbon dioxide.",
        "the oceans are home to millions of species of plants and animals.",
    ],

    "Rocks and Minerals": [
        "rocks are solid materials made of one or more minerals.",
        "minerals are natural, solid substances with a specific chemical composition.",
        "igneous rocks form when magma cools and hardens.",
        "when magma cools slowly underground, it forms large crystals.",
        "when lava cools quickly at the surface, it forms small crystals.",
        "sedimentary rocks form from layers of sediment that are compressed over time.",
        "rivers carry sediment such as sand and mud to the sea.",
        "layers of sediment build up on the sea floor.",
        "over millions of years, pressure from above hardens the sediment into rock.",
        "metamorphic rocks form when existing rocks are changed by heat and pressure.",
        "high temperatures and pressures deep in the earth transform the structure of rocks.",
        "the rock cycle describes how rocks are continuously transformed from one type to another.",
    ],

}


# ─────────────────────────────────────────────
# SENTENCE CLEANER
# ─────────────────────────────────────────────

def clean(sentence: str) -> str:
    s = sentence.strip().lower()
    if not s.endswith('.'):
        s += '.'
    return s


# ─────────────────────────────────────────────
# BUILD OUTPUT FILES
# ─────────────────────────────────────────────

def build(output_dir: str = "/tmp/data"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sentences_path = out / "sentences.txt"
    pairs_path = out / "pairs.jsonl"

    all_sentences = []
    all_pairs = []

    for title, raw_sentences in ARTICLES.items():
        sentences = [clean(s) for s in raw_sentences]

        for s in sentences:
            all_sentences.append(s)

        for i in range(len(sentences) - 1):
            all_pairs.append({
                "input": sentences[i],
                "target": sentences[i + 1],
                "article": title,
                "pair_idx": i
            })

    with open(sentences_path, "w") as f:
        for s in all_sentences:
            f.write(s + "\n")

    with open(pairs_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    stats = {
        "articles": len(ARTICLES),
        "total_sentences": len(all_sentences),
        "total_pairs": len(all_pairs),
        "domains": ["weather/atmosphere", "physics", "biology", "earth/geology"],
    }

    print("=== Corpus Built ===")
    print(json.dumps(stats, indent=2))
    print(f"\nsentences.txt → {sentences_path}")
    print(f"pairs.jsonl   → {pairs_path}")

    return stats


if __name__ == "__main__":
    build()