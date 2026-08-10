"""
syllabus.py — Complete Structured Syllabus Class 1-12.

Every topic is an atomic teaching unit.
Organised by subject, grade, topic in correct pedagogical order.
Source: NCERT + Common Core + NGSS aligned.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Topic:
    subject: str
    grade:   int
    name:    str
    description: str = ''


def _t(subject, grade, name, desc=''):
    return Topic(subject=subject, grade=grade, name=name, description=desc)


SYLLABUS: List[Topic] = [

    # ── MATHEMATICS ───────────────────────────────────────────────

    # Class 1
    _t('Mathematics', 1, 'Shapes and Spatial Understanding',   'Basic 2D shapes, positions, directions'),
    _t('Mathematics', 1, 'Numbers 1 to 9',                     'Counting, writing, comparing single digits'),
    _t('Mathematics', 1, 'Addition up to 9',                   'Combining small groups of objects'),
    _t('Mathematics', 1, 'Subtraction up to 9',                'Taking away, difference between numbers'),
    _t('Mathematics', 1, 'Numbers 10 to 20',                   'Teen numbers, place value introduction'),
    _t('Mathematics', 1, 'Time',                               'Days of week, morning afternoon evening'),
    _t('Mathematics', 1, 'Measurement',                        'Longer shorter heavier lighter'),
    _t('Mathematics', 1, 'Numbers 21 to 99',                   'Tens and ones, counting groups'),
    _t('Mathematics', 1, 'Patterns',                           'Repeating patterns, what comes next'),

    # Class 2
    _t('Mathematics', 2, 'Numbers up to 100',                  'Place value tens and ones, ordering'),
    _t('Mathematics', 2, 'Addition with Carrying',             'Adding two digit numbers with regrouping'),
    _t('Mathematics', 2, 'Subtraction with Borrowing',         'Subtracting two digit numbers with regrouping'),
    _t('Mathematics', 2, 'Multiplication Introduction',        'Equal groups, repeated addition'),
    _t('Mathematics', 2, 'Division Introduction',              'Sharing equally, grouping'),
    _t('Mathematics', 2, 'Measurement of Length',              'Centimetres and metres'),
    _t('Mathematics', 2, 'Money',                              'Coins and notes, simple transactions'),
    _t('Mathematics', 2, 'Data Handling',                      'Tally marks, simple pictographs'),

    # Class 3
    _t('Mathematics', 3, 'Four Digit Numbers',                 'Thousands hundreds tens ones'),
    _t('Mathematics', 3, 'Addition and Subtraction',           'Four digit operations with carrying borrowing'),
    _t('Mathematics', 3, 'Multiplication Tables',              'Tables 2 to 10, properties of multiplication'),
    _t('Mathematics', 3, 'Division',                           'Division as equal sharing and grouping'),
    _t('Mathematics', 3, 'Fractions Introduction',             'Half quarter three quarters, equal parts'),
    _t('Mathematics', 3, 'Geometry Basics',                    'Lines angles triangles rectangles circles'),
    _t('Mathematics', 3, 'Perimeter',                          'Measuring boundary of shapes'),
    _t('Mathematics', 3, 'Time and Calendar',                  'Hours minutes, reading clocks, calendar'),

    # Class 4
    _t('Mathematics', 4, 'Large Numbers',                      'Numbers up to 10000, place value'),
    _t('Mathematics', 4, 'Factors and Multiples',              'Factors divisors multiples, prime and composite'),
    _t('Mathematics', 4, 'Fractions',                          'Equivalent fractions, comparing, adding subtracting'),
    _t('Mathematics', 4, 'Decimals Introduction',              'Tenths hundredths, decimal notation'),
    _t('Mathematics', 4, 'Geometry',                           'Angles, types of triangles, quadrilaterals'),
    _t('Mathematics', 4, 'Area',                               'Area of rectangles and squares by counting'),
    _t('Mathematics', 4, 'Symmetry',                           'Line of symmetry, mirror images'),

    # Class 5
    _t('Mathematics', 5, 'Large Number Operations',            'Multiply and divide large numbers'),
    _t('Mathematics', 5, 'Fractions and Decimals',             'Operations with fractions and decimals'),
    _t('Mathematics', 5, 'Percentage Introduction',            'Percent as part of hundred'),
    _t('Mathematics', 5, 'Average',                            'Finding average of a set of numbers'),
    _t('Mathematics', 5, 'Area and Volume',                    'Area of triangles, volume of cuboid'),
    _t('Mathematics', 5, 'Simple Equations Introduction',      'Finding unknown in simple equations'),

    # Class 6
    _t('Mathematics', 6, 'Knowing Our Numbers',                'Comparing ordering large numbers, estimation'),
    _t('Mathematics', 6, 'Whole Numbers',                      'Properties, number line, patterns'),
    _t('Mathematics', 6, 'Playing with Numbers',               'Divisibility tests, HCF, LCM'),
    _t('Mathematics', 6, 'Basic Geometrical Ideas',            'Points lines rays angles curves polygons'),
    _t('Mathematics', 6, 'Understanding Elementary Shapes',    'Measuring angles, 3D shapes'),
    _t('Mathematics', 6, 'Integers',                           'Negative numbers, number line, operations'),
    _t('Mathematics', 6, 'Fractions',                          'Types, operations, simplification'),
    _t('Mathematics', 6, 'Decimals',                           'Operations, conversion, applications'),
    _t('Mathematics', 6, 'Data Handling',                      'Bar graphs, mean median mode'),
    _t('Mathematics', 6, 'Mensuration',                        'Perimeter and area of basic shapes'),
    _t('Mathematics', 6, 'Algebra Introduction',               'Variables, expressions, simple equations'),
    _t('Mathematics', 6, 'Ratio and Proportion',               'Ratio, equivalent ratios, proportion'),

    # Class 7
    _t('Mathematics', 7, 'Integers',                           'Operations, properties, applications'),
    _t('Mathematics', 7, 'Fractions and Decimals',             'Multiplication division of fractions decimals'),
    _t('Mathematics', 7, 'Simple Equations',                   'Solving equations, applications'),
    _t('Mathematics', 7, 'Lines and Angles',                   'Complementary supplementary parallel lines'),
    _t('Mathematics', 7, 'Triangles and Properties',           'Angle sum, exterior angle, Pythagoras'),
    _t('Mathematics', 7, 'Congruence of Triangles',            'SSS SAS ASA RHS conditions'),
    _t('Mathematics', 7, 'Comparing Quantities',               'Ratio proportion percent profit loss'),
    _t('Mathematics', 7, 'Rational Numbers',                   'Properties, number line, operations'),
    _t('Mathematics', 7, 'Perimeter and Area',                 'Circles, composite shapes'),
    _t('Mathematics', 7, 'Algebraic Expressions',              'Like unlike terms, addition subtraction'),
    _t('Mathematics', 7, 'Exponents and Powers',               'Laws of exponents, standard form'),

    # Class 8
    _t('Mathematics', 8, 'Rational Numbers',                   'Properties, between two rationals'),
    _t('Mathematics', 8, 'Linear Equations in One Variable',   'Solving, applications, word problems'),
    _t('Mathematics', 8, 'Understanding Quadrilaterals',       'Properties, parallelogram, rhombus, trapezium'),
    _t('Mathematics', 8, 'Squares and Square Roots',           'Perfect squares, finding square roots'),
    _t('Mathematics', 8, 'Cubes and Cube Roots',               'Perfect cubes, finding cube roots'),
    _t('Mathematics', 8, 'Comparing Quantities',               'Compound interest, discount, tax'),
    _t('Mathematics', 8, 'Algebraic Expressions and Identities','Multiplication, standard identities'),
    _t('Mathematics', 8, 'Mensuration',                        'Area of polygons, surface area, volume'),
    _t('Mathematics', 8, 'Direct and Inverse Proportion',      'Applications and word problems'),
    _t('Mathematics', 8, 'Factorisation',                      'Common factors, regrouping, identities'),

    # Class 9
    _t('Mathematics', 9, 'Number Systems',                     'Irrational numbers, real number line, surds'),
    _t('Mathematics', 9, 'Polynomials',                        'Zeroes, remainder theorem, factor theorem'),
    _t('Mathematics', 9, 'Coordinate Geometry',                'Cartesian plane, plotting points, quadrants'),
    _t('Mathematics', 9, 'Linear Equations in Two Variables',  'Graph of linear equation, solutions'),
    _t('Mathematics', 9, 'Lines and Angles',                   'Axioms, theorems, parallel lines transversal'),
    _t('Mathematics', 9, 'Triangles',                          'Congruence criteria, properties, inequalities'),
    _t('Mathematics', 9, 'Quadrilaterals',                     'Properties, mid-point theorem'),
    _t('Mathematics', 9, 'Circles',                            'Chord, arc, angle subtended theorems'),
    _t('Mathematics', 9, 'Herons Formula',                     'Area of triangle using three sides'),
    _t('Mathematics', 9, 'Surface Areas and Volumes',          'Cuboid cylinder cone sphere'),
    _t('Mathematics', 9, 'Statistics',                         'Mean median mode, grouped data'),
    _t('Mathematics', 9, 'Probability',                        'Basic probability, experiments, events'),

    # Class 10
    _t('Mathematics', 10, 'Real Numbers',                      'Euclids algorithm, irrational numbers, decimal expansion'),
    _t('Mathematics', 10, 'Polynomials',                       'Geometric meaning of zeroes, relationship zeroes coefficients'),
    _t('Mathematics', 10, 'Pair of Linear Equations',          'Graphical algebraic solutions, applications'),
    _t('Mathematics', 10, 'Quadratic Equations',               'Factorisation, formula, discriminant, applications'),
    _t('Mathematics', 10, 'Arithmetic Progressions',           'nth term, sum of n terms, applications'),
    _t('Mathematics', 10, 'Triangles',                         'Similarity criteria, basic proportionality, Pythagoras'),
    _t('Mathematics', 10, 'Coordinate Geometry',               'Distance formula, section formula, area of triangle'),
    _t('Mathematics', 10, 'Introduction to Trigonometry',      'Ratios, identities, complementary angles'),
    _t('Mathematics', 10, 'Applications of Trigonometry',      'Heights and distances, angles of elevation depression'),
    _t('Mathematics', 10, 'Circles',                           'Tangent, number of tangents from external point'),
    _t('Mathematics', 10, 'Areas Related to Circles',          'Area of sector segment, combination of plane figures'),
    _t('Mathematics', 10, 'Surface Areas and Volumes',         'Combinations of solids, conversion of solids'),
    _t('Mathematics', 10, 'Statistics',                        'Mean median mode of grouped data, cumulative frequency'),
    _t('Mathematics', 10, 'Probability',                       'Classical probability, complementary events'),

    # Class 11
    _t('Mathematics', 11, 'Sets',                              'Types, operations, Venn diagrams, applications'),
    _t('Mathematics', 11, 'Relations and Functions',           'Domain range, types of functions, graphs'),
    _t('Mathematics', 11, 'Trigonometric Functions',           'Radian measure, unit circle, graphs, identities'),
    _t('Mathematics', 11, 'Mathematical Induction',            'Principle, proving statements'),
    _t('Mathematics', 11, 'Complex Numbers',                   'Algebra, modulus argument, Argand plane'),
    _t('Mathematics', 11, 'Linear Inequalities',               'Solving, graphical representation, systems'),
    _t('Mathematics', 11, 'Permutations and Combinations',     'Counting principle, nPr, nCr, applications'),
    _t('Mathematics', 11, 'Binomial Theorem',                  'Expansion, general term, middle term'),
    _t('Mathematics', 11, 'Sequences and Series',              'AP, GP, sum formulas, AM GM'),
    _t('Mathematics', 11, 'Straight Lines',                    'Slope, equations of line, distance, family of lines'),
    _t('Mathematics', 11, 'Conic Sections',                    'Circle ellipse parabola hyperbola, standard forms'),
    _t('Mathematics', 11, 'Three Dimensional Geometry',        'Coordinates, distance, section formula'),
    _t('Mathematics', 11, 'Limits and Derivatives',            'Intuitive limit, algebra of limits, derivative'),
    _t('Mathematics', 11, 'Statistics',                        'Measures of dispersion, variance, SD, CV'),
    _t('Mathematics', 11, 'Probability',                       'Random experiments, axiomatic approach, theorems'),

    # Class 12
    _t('Mathematics', 12, 'Relations and Functions',           'Composition, invertible functions, binary operations'),
    _t('Mathematics', 12, 'Inverse Trigonometric Functions',   'Domain range, properties, identities'),
    _t('Mathematics', 12, 'Matrices',                          'Types, operations, transpose, elementary operations'),
    _t('Mathematics', 12, 'Determinants',                      'Properties, cofactors, adjoint, inverse, Cramers rule'),
    _t('Mathematics', 12, 'Continuity and Differentiability',  'Chain rule, implicit, logarithmic, parametric, Rolles MVT'),
    _t('Mathematics', 12, 'Application of Derivatives',        'Rate of change, tangents normals, maxima minima'),
    _t('Mathematics', 12, 'Integrals',                         'Indefinite definite, methods, properties'),
    _t('Mathematics', 12, 'Application of Integrals',          'Area under curves, between two curves'),
    _t('Mathematics', 12, 'Differential Equations',            'Order degree, formation, variable separable, homogeneous'),
    _t('Mathematics', 12, 'Vector Algebra',                    'Types, operations, dot cross product, scalar triple'),
    _t('Mathematics', 12, 'Three Dimensional Geometry',        'Direction cosines, lines planes, angle between'),
    _t('Mathematics', 12, 'Linear Programming',                'Formulation, graphical method, corner point'),
    _t('Mathematics', 12, 'Probability',                       'Conditional, multiplication theorem, Bayes, distributions'),

    # ── SCIENCE Class 1-8 ─────────────────────────────────────────

    _t('Science', 1, 'Living and Non-living Things',           'Differences, examples, characteristics of life'),
    _t('Science', 1, 'Plants Around Us',                       'Parts of plant, needs of plants, types'),
    _t('Science', 1, 'Animals Around Us',                      'Types, homes, food, domestic and wild'),
    _t('Science', 1, 'Our Body',                               'Body parts and their functions, senses'),
    _t('Science', 1, 'Food We Eat',                            'Types of food, healthy eating'),
    _t('Science', 1, 'Water',                                  'Uses of water, sources, save water'),
    _t('Science', 1, 'Air Around Us',                          'Air everywhere, uses, cannot be seen'),
    _t('Science', 1, 'Weather and Seasons',                    'Hot cold rainy, seasonal changes'),

    _t('Science', 2, 'Plants and Their Parts',                 'Root stem leaf flower fruit, their functions'),
    _t('Science', 2, 'Animals and Their Homes',                'Nest burrow stable web, habitat'),
    _t('Science', 2, 'Food and Nutrition',                     'Fruits vegetables grains, balanced diet'),
    _t('Science', 2, 'Water Cycle',                            'Evaporation condensation rainfall, importance'),
    _t('Science', 2, 'Air and Weather',                        'Properties of air, wind, weather changes'),
    _t('Science', 2, 'Our Environment',                        'Keeping surroundings clean, waste disposal'),

    _t('Science', 3, 'Plants and Growth',                      'Germination, growth conditions, photosynthesis introduction'),
    _t('Science', 3, 'Animals and Adaptation',                 'Adaptation to habitat, migration hibernation'),
    _t('Science', 3, 'Food Chain',                             'Producer consumer decomposer, simple food chains'),
    _t('Science', 3, 'States of Matter',                       'Solid liquid gas, properties, changes of state'),
    _t('Science', 3, 'Light and Shadow',                       'Sources of light, transparent opaque, shadows'),
    _t('Science', 3, 'Forces and Motion',                      'Push pull, types of motion, friction'),
    _t('Science', 3, 'Simple Machines',                        'Lever pulley wheel axle inclined plane'),

    _t('Science', 4, 'Plants Reproduction',                    'Seeds fruits, vegetative propagation, dispersal'),
    _t('Science', 4, 'Animals and Life Cycles',                'Life cycle of frog butterfly insect'),
    _t('Science', 4, 'Human Digestive System',                 'Mouth stomach intestines, digestion process'),
    _t('Science', 4, 'Rocks and Soil',                         'Types of rocks, soil formation, soil types'),
    _t('Science', 4, 'Electricity Basics',                     'Electric circuit, conductors insulators, simple circuits'),
    _t('Science', 4, 'Magnets',                                'Properties, poles, attraction repulsion, uses'),
    _t('Science', 4, 'Natural Disasters',                      'Earthquakes floods cyclones, safety measures'),

    _t('Science', 5, 'Photosynthesis',                         'Process, chlorophyll, light water CO2, oxygen release'),
    _t('Science', 5, 'Ecosystems',                             'Biotic abiotic, food web, energy flow'),
    _t('Science', 5, 'Human Body Systems',                     'Skeletal muscular circulatory respiratory'),
    _t('Science', 5, 'Weather and Climate',                    'Difference, climate zones, climate change basics'),
    _t('Science', 5, 'Sound',                                  'Vibration, properties of sound, hearing'),
    _t('Science', 5, 'Materials and Properties',               'Natural synthetic, properties, uses'),

    _t('Science', 6, 'Food Sources and Components',            'Macronutrients micronutrients, deficiency diseases'),
    _t('Science', 6, 'Fibre to Fabric',                        'Natural synthetic fibres, weaving spinning'),
    _t('Science', 6, 'Separation of Substances',               'Methods of separation, sieving filtration evaporation'),
    _t('Science', 6, 'Changes Around Us',                      'Reversible irreversible, physical chemical changes'),
    _t('Science', 6, 'Getting to Know Plants',                 'Root stem leaf venation, transpiration'),
    _t('Science', 6, 'Body Movements',                         'Types of joints, muscles, earthworm movement'),
    _t('Science', 6, 'The Living Organisms and Surroundings',  'Habitat adaptation biotic abiotic components'),
    _t('Science', 6, 'Motion and Measurement',                 'Types of motion, SI units, measuring length'),
    _t('Science', 6, 'Light Shadows and Reflection',           'Rectilinear propagation, pinhole camera, mirrors'),
    _t('Science', 6, 'Electricity and Circuits',               'Electric cell, switch, closed open circuit'),
    _t('Science', 6, 'Fun with Magnets',                       'Properties, compass, magnetic and non-magnetic'),
    _t('Science', 6, 'Water',                                  'Sources, water cycle, conservation, groundwater'),
    _t('Science', 6, 'Air Around Us',                          'Composition, role in combustion, atmosphere'),
    _t('Science', 6, 'Garbage Management',                     'Waste types, composting, reduce reuse recycle'),

    _t('Science', 7, 'Nutrition in Plants',                    'Autotrophs heterotrophs, photosynthesis, nitrogen cycle'),
    _t('Science', 7, 'Nutrition in Animals',                   'Digestion in humans amoeba, absorption'),
    _t('Science', 7, 'Heat',                                   'Temperature, conduction convection radiation, expansion'),
    _t('Science', 7, 'Acids Bases and Salts',                  'Properties, indicators, neutralisation, everyday applications'),
    _t('Science', 7, 'Physical and Chemical Changes',          'Rusting burning dissolving, characteristics'),
    _t('Science', 7, 'Respiration in Organisms',               'Aerobic anaerobic, breathing in animals plants'),
    _t('Science', 7, 'Transportation in Plants and Animals',   'Xylem phloem, blood circulation, heart'),
    _t('Science', 7, 'Reproduction in Plants',                 'Asexual sexual, pollination fertilisation, seed dispersal'),
    _t('Science', 7, 'Motion and Time',                        'Speed distance time, uniform non-uniform motion, graphs'),
    _t('Science', 7, 'Electric Current and Effects',           'Heating effect magnetic effect, electromagnet'),
    _t('Science', 7, 'Light',                                  'Reflection laws, plane mirrors, concave convex, eye'),
    _t('Science', 7, 'Forests',                                'Importance, canopy understory, forest products, conservation'),
    _t('Science', 7, 'Water Management',                       'Floods droughts, rain water harvesting, depletion'),

    _t('Science', 8, 'Crop Production and Management',         'Kharif rabi crops, agricultural practices, irrigation'),
    _t('Science', 8, 'Microorganisms',                         'Types, useful harmful, food preservation, antibiotics'),
    _t('Science', 8, 'Conservation of Plants and Animals',     'Deforestation, biodiversity, wildlife sanctuaries'),
    _t('Science', 8, 'Cell Structure and Functions',           'Cell theory, plant animal cell, organelles'),
    _t('Science', 8, 'Reproduction in Animals',               'Sexual asexual, internal external fertilisation'),
    _t('Science', 8, 'Reaching Age of Adolescence',            'Puberty, hormones, reproductive health, nutrition'),
    _t('Science', 8, 'Force and Pressure',                     'Contact non-contact forces, pressure, atmospheric pressure'),
    _t('Science', 8, 'Friction',                               'Types, factors affecting, advantages disadvantages'),
    _t('Science', 8, 'Sound',                                  'Vibration, characteristics, noise pollution, range'),
    _t('Science', 8, 'Chemical Effects of Electric Current',   'Electrolysis, electroplating, conductivity of liquids'),
    _t('Science', 8, 'Natural Phenomena',                      'Lightning, earthquakes, causes effects precautions'),
    _t('Science', 8, 'Natural Resources',                      'Conservation, sustainable use, coal petroleum'),

    # ── PHYSICS Class 9-12 ────────────────────────────────────────

    _t('Physics', 9, 'Motion',                                  'Distance displacement, speed velocity, acceleration, graphs'),
    _t('Physics', 9, 'Force and Newtons Laws',                  'Three laws, inertia, momentum, action reaction'),
    _t('Physics', 9, 'Gravitation',                             'Universal law, acceleration due to gravity, weight, pressure'),
    _t('Physics', 9, 'Work Energy and Power',                   'Work done, kinetic potential energy, conservation, power'),
    _t('Physics', 9, 'Sound',                                   'Wave nature, characteristics, SONAR, range of hearing'),

    _t('Physics', 10, 'Light Reflection and Refraction',        'Laws, spherical mirrors, lens formula, power of lens'),
    _t('Physics', 10, 'Human Eye and Defects',                  'Structure, accommodation, myopia hypermetropia, dispersion'),
    _t('Physics', 10, 'Electricity',                            'Charge current resistance, Ohms law, circuit, power'),
    _t('Physics', 10, 'Magnetic Effects of Current',            'Magnetic field, force on conductor, AC DC, motor generator'),
    _t('Physics', 10, 'Sources of Energy',                      'Conventional non-conventional, renewable, fossil fuels'),

    _t('Physics', 11, 'Units and Measurement',                  'SI units, significant figures, dimensional analysis'),
    _t('Physics', 11, 'Kinematics',                             'Equations of motion, projectile, circular motion'),
    _t('Physics', 11, 'Laws of Motion',                         'Newtons laws, friction, circular motion dynamics'),
    _t('Physics', 11, 'Work Energy and Power',                  'Work-energy theorem, conservation, elastic inelastic collision'),
    _t('Physics', 11, 'Rotational Motion',                      'Torque, angular momentum, moment of inertia, rolling'),
    _t('Physics', 11, 'Gravitation',                            'Keplers laws, orbital velocity, escape velocity, satellites'),
    _t('Physics', 11, 'Properties of Matter',                   'Elasticity, viscosity, surface tension, Bernoulli'),
    _t('Physics', 11, 'Thermodynamics',                         'Zeroth first second laws, heat engines, Carnot cycle'),
    _t('Physics', 11, 'Kinetic Theory of Gases',                'Assumptions, pressure, temperature, specific heats'),
    _t('Physics', 11, 'Oscillations',                           'SHM, energy, damped forced oscillations, resonance'),
    _t('Physics', 11, 'Waves',                                  'Transverse longitudinal, superposition, standing waves, Doppler'),

    _t('Physics', 12, 'Electric Charges and Fields',            'Coulombs law, electric field, flux, Gauss law'),
    _t('Physics', 12, 'Electrostatic Potential and Capacitance','Potential, capacitor, dielectric, energy stored'),
    _t('Physics', 12, 'Current Electricity',                    'Drift velocity, resistivity, Kirchhoffs laws, Wheatstone'),
    _t('Physics', 12, 'Moving Charges and Magnetism',           'Biot-Savart, Amperes law, force on conductor, cyclotron'),
    _t('Physics', 12, 'Magnetism and Matter',                   'Dipole, earths magnetism, para dia ferro magnetic materials'),
    _t('Physics', 12, 'Electromagnetic Induction',              'Faradays laws, Lenz, self mutual inductance, eddy currents'),
    _t('Physics', 12, 'Alternating Current',                    'RMS, phasor, LCR circuit, resonance, transformer'),
    _t('Physics', 12, 'Electromagnetic Waves',                  'Maxwells equations, spectrum, properties of EM waves'),
    _t('Physics', 12, 'Ray Optics',                             'Reflection refraction, total internal reflection, lens mirror'),
    _t('Physics', 12, 'Wave Optics',                            'Huygens, interference, diffraction, polarisation'),
    _t('Physics', 12, 'Dual Nature of Radiation',               'Photoelectric effect, de Broglie, Davisson-Germer'),
    _t('Physics', 12, 'Atoms',                                  'Rutherford Bohr model, spectral series, energy levels'),
    _t('Physics', 12, 'Nuclei',                                 'Nuclear force, binding energy, radioactivity, fission fusion'),
    _t('Physics', 12, 'Semiconductor Electronics',              'Energy bands, p-n junction, diode transistor, logic gates'),

    # ── CHEMISTRY Class 9-12 ─────────────────────────────────────

    _t('Chemistry', 9, 'Matter in Our Surroundings',            'States, properties, evaporation, sublimation, plasma BEC'),
    _t('Chemistry', 9, 'Is Matter Around Us Pure',              'Mixture solution compound, separation methods, colloids'),
    _t('Chemistry', 9, 'Atoms and Molecules',                   'Laws of chemical combination, atomic mass, mole concept'),
    _t('Chemistry', 9, 'Structure of the Atom',                 'Thomson Rutherford Bohr model, electrons protons neutrons'),

    _t('Chemistry', 10, 'Chemical Reactions and Equations',     'Types, balancing, combination decomposition displacement'),
    _t('Chemistry', 10, 'Acids Bases and Salts',                'Properties, pH, neutralisation, salts preparation'),
    _t('Chemistry', 10, 'Metals and Non-metals',                'Physical chemical properties, reactivity series, corrosion'),
    _t('Chemistry', 10, 'Carbon and Its Compounds',             'Covalent bonding, allotropes, organic compounds, soap'),
    _t('Chemistry', 10, 'Periodic Classification of Elements',  'Newlands Mendeleev Modern periodic table, trends'),

    _t('Chemistry', 11, 'Basic Concepts of Chemistry',          'Mole, empirical molecular formula, stoichiometry'),
    _t('Chemistry', 11, 'Structure of Atom',                    'Quantum mechanical model, orbitals, quantum numbers'),
    _t('Chemistry', 11, 'Classification of Elements',           'Modern periodic table, trends, anomalies'),
    _t('Chemistry', 11, 'Chemical Bonding',                     'Ionic covalent metallic, VSEPR, hybridisation, MO theory'),
    _t('Chemistry', 11, 'States of Matter',                     'Kinetic theory, gas laws, liquid state, surface tension'),
    _t('Chemistry', 11, 'Chemical Thermodynamics',              'System surroundings, enthalpy entropy, Gibbs energy'),
    _t('Chemistry', 11, 'Equilibrium',                          'Chemical ionic equilibrium, Le Chateliers, pH, buffers'),
    _t('Chemistry', 11, 'Redox Reactions',                      'Oxidation reduction, oxidation state, balancing'),
    _t('Chemistry', 11, 'Hydrogen',                             'Properties, hydrides, water, heavy water, hydrogen economy'),
    _t('Chemistry', 11, 'The s-Block Elements',                 'Alkali alkaline earth metals, compounds, uses'),
    _t('Chemistry', 11, 'The p-Block Elements Group 13-14',     'Boron carbon family, allotropes, compounds'),
    _t('Chemistry', 11, 'Organic Chemistry Basics',             'Nomenclature, isomerism, reaction mechanisms'),
    _t('Chemistry', 11, 'Hydrocarbons',                         'Alkanes alkenes alkynes, aromatic, reactions'),

    _t('Chemistry', 12, 'Solid State',                          'Crystal structure, unit cell, defects, properties'),
    _t('Chemistry', 12, 'Solutions',                            'Types, colligative properties, Raoult, Vant Hoff'),
    _t('Chemistry', 12, 'Electrochemistry',                     'Galvanic cell, Nernst equation, electrolysis, batteries'),
    _t('Chemistry', 12, 'Chemical Kinetics',                    'Rate, rate law, order, Arrhenius, mechanisms'),
    _t('Chemistry', 12, 'Surface Chemistry',                    'Adsorption, catalysis, colloids, emulsions'),
    _t('Chemistry', 12, 'd and f Block Elements',               'Properties, electronic configuration, compounds, interstitial'),
    _t('Chemistry', 12, 'Coordination Compounds',               'Nomenclature, bonding, isomerism, stability'),
    _t('Chemistry', 12, 'Haloalkanes and Haloarenes',           'Nomenclature, preparation, reactions, uses'),
    _t('Chemistry', 12, 'Alcohols Phenols and Ethers',          'Preparation, properties, reactions, uses'),
    _t('Chemistry', 12, 'Aldehydes and Ketones',                'Preparation, reactions, nucleophilic addition'),
    _t('Chemistry', 12, 'Amines',                               'Classification, preparation, properties, dyes'),
    _t('Chemistry', 12, 'Biomolecules',                         'Carbohydrates proteins lipids nucleic acids vitamins'),
    _t('Chemistry', 12, 'Polymers',                             'Addition condensation, natural synthetic, rubber plastics'),

    # ── BIOLOGY Class 9-12 ────────────────────────────────────────

    _t('Biology', 9, 'The Fundamental Unit of Life',            'Cell theory, prokaryotic eukaryotic, organelles'),
    _t('Biology', 9, 'Tissues',                                 'Plant animal tissues, types, structure function'),
    _t('Biology', 9, 'Diversity in Living Organisms',           'Classification kingdoms, nomenclature, five kingdom'),
    _t('Biology', 9, 'Why Do We Fall Ill',                      'Health disease, causes, infectious non-infectious, immunity'),
    _t('Biology', 9, 'Natural Resources',                       'Air water soil, nitrogen carbon cycle, ozone'),
    _t('Biology', 9, 'Improvement in Food Resources',           'Crop improvement, animal husbandry, aquaculture'),

    _t('Biology', 10, 'Life Processes',                         'Nutrition respiration transportation excretion in living things'),
    _t('Biology', 10, 'Control and Coordination',               'Nervous system, brain, reflex arc, hormones, endocrine'),
    _t('Biology', 10, 'Reproduction',                           'Asexual sexual, human reproduction, contraception, STI'),
    _t('Biology', 10, 'Heredity and Evolution',                 'Mendels laws, sex determination, evolution, natural selection'),
    _t('Biology', 10, 'Our Environment',                        'Ecosystem, food chain web, ozone depletion, waste management'),
    _t('Biology', 10, 'Management of Natural Resources',        'Forest water coal petroleum, sustainable development'),

    _t('Biology', 11, 'The Living World',                       'Taxonomy, classification, nomenclature, keys'),
    _t('Biology', 11, 'Biological Classification',              'Five kingdoms, characteristics, viruses lichens'),
    _t('Biology', 11, 'Plant Kingdom',                          'Algae bryophytes pteridophytes gymnosperms angiosperms'),
    _t('Biology', 11, 'Animal Kingdom',                         'Basis of classification, phyla, characteristics'),
    _t('Biology', 11, 'Morphology of Flowering Plants',         'Root stem leaf flower fruit seed, modifications'),
    _t('Biology', 11, 'Anatomy of Flowering Plants',            'Tissues, meristematic permanent, anatomy of organs'),
    _t('Biology', 11, 'Cell Biology and Biomolecules',          'Cell cycle, mitosis meiosis, biomolecules functions'),
    _t('Biology', 11, 'Photosynthesis in Higher Plants',        'Light dark reactions, C3 C4 CAM, factors affecting'),
    _t('Biology', 11, 'Respiration in Plants',                  'Glycolysis, Krebs cycle, electron transport, fermentation'),
    _t('Biology', 11, 'Plant Growth and Development',           'Growth phases, PGRs, seed dormancy, photoperiodism'),
    _t('Biology', 11, 'Digestion and Absorption',               'Alimentary canal, enzymes, absorption, disorders'),
    _t('Biology', 11, 'Breathing and Exchange of Gases',        'Lungs, mechanism, transport, disorders'),
    _t('Biology', 11, 'Body Fluids and Circulation',            'Blood lymph, heart, cardiac cycle, disorders'),
    _t('Biology', 11, 'Excretion',                              'Human excretory system, urine formation, disorders, plants'),
    _t('Biology', 11, 'Locomotion and Movement',                'Types of movement, skeletal muscle, joints, disorders'),
    _t('Biology', 11, 'Neural Control and Coordination',        'Neuron, CNS PNS, reflex, sense organs'),
    _t('Biology', 11, 'Chemical Coordination',                  'Endocrine glands, hormones, mechanism of action'),

    _t('Biology', 12, 'Reproduction in Organisms',              'Modes of reproduction, life span, asexual types'),
    _t('Biology', 12, 'Sexual Reproduction in Flowering Plants','Flower structure, pollination, fertilisation, seed fruit'),
    _t('Biology', 12, 'Human Reproduction',                     'Male female reproductive system, gametogenesis, fertilisation'),
    _t('Biology', 12, 'Reproductive Health',                    'Population, birth control, STI, assisted reproductive'),
    _t('Biology', 12, 'Principles of Inheritance',              'Mendels laws, chromosomal theory, sex-linked, pedigree'),
    _t('Biology', 12, 'Molecular Basis of Inheritance',         'DNA structure, replication, transcription, translation, genetic code'),
    _t('Biology', 12, 'Evolution',                              'Origin of life, evidence, Darwinism, speciation, Hardy-Weinberg'),
    _t('Biology', 12, 'Human Health and Disease',               'Innate adaptive immunity, AIDS, cancer, drugs alcohol'),
    _t('Biology', 12, 'Biotechnology Principles',               'Recombinant DNA, tools, PCR, gel electrophoresis'),
    _t('Biology', 12, 'Biotechnology Applications',             'GM organisms, insulin BT cotton, gene therapy, bioethics'),
    _t('Biology', 12, 'Ecology and Ecosystems',                 'Organisms populations communities, energy flow, nutrient cycling'),
    _t('Biology', 12, 'Biodiversity and Conservation',          'Patterns, loss, conservation strategies, hotspots'),
    _t('Biology', 12, 'Environmental Issues',                   'Pollution types, greenhouse effect, depletion, case studies'),

    # ── ENGLISH Class 1-12 ────────────────────────────────────────

    _t('English', 1,  'Alphabet and Phonics',                   'Letter sounds, phonemic awareness, CVC words'),
    _t('English', 2,  'Reading Simple Stories',                  'Comprehension, main idea, characters, sequence'),
    _t('English', 3,  'Nouns and Pronouns',                     'Common proper, singular plural, personal pronouns'),
    _t('English', 4,  'Adjectives and Adverbs',                 'Descriptive words, degrees of comparison'),
    _t('English', 5,  'Tenses and Sentences',                   'Simple present past future, sentence types'),
    _t('English', 6,  'Parts of Speech',                        'All eight parts, usage in sentences'),
    _t('English', 7,  'Tenses Advanced',                        'Perfect progressive tenses, time expressions'),
    _t('English', 8,  'Active and Passive Voice',               'Transformation, when to use, applications'),
    _t('English', 9,  'Literature Prose Analysis',              'Theme character setting plot, textual evidence'),
    _t('English', 9,  'Poetry Analysis',                        'Figurative language, rhyme rhythm, tone, message'),
    _t('English', 10, 'Grammar Mastery',                        'Reported speech, clauses, punctuation, idioms'),
    _t('English', 10, 'Writing Skills',                         'Letter essay article report paragraph formal informal'),
    _t('English', 11, 'Advanced Reading',                       'Inference critical reading, skimming scanning'),
    _t('English', 11, 'Advanced Writing',                       'Argumentative discursive analytical essays'),
    _t('English', 12, 'Literature Analysis',                    'Critical appreciation, symbolism, context, comparison'),
    _t('English', 12, 'Communication Skills',                   'Oral presentation, discussion, interview, précis'),

    # ── HISTORY Class 6-12 ────────────────────────────────────────

    _t('History', 6,  'Earliest Humans',                        'Hunter gatherers, tools, rock art, settled communities'),
    _t('History', 6,  'First Cities',                           'Harappan civilisation, urban planning, trade, decline'),
    _t('History', 6,  'Early Kingdoms',                         'Janapadas mahajanapadas, republics, Magadha'),
    _t('History', 6,  'New Religious Ideas',                    'Buddhism Jainism, Upanishads, spread, influence'),
    _t('History', 6,  'Mauryan Empire',                         'Chandragupta, Ashoka, edicts, administration, decline'),
    _t('History', 7,  'Medieval Kingdoms',                      'Pallavas Chalukyas Rashtrakutas, architecture, literature'),
    _t('History', 7,  'Delhi Sultanate',                        'Establishment, rulers, administration, culture'),
    _t('History', 7,  'Mughal Empire',                          'Akbar to Aurangzeb, administration, art, decline'),
    _t('History', 7,  'Bhakti and Sufi Movements',              'Saints, teachings, impact on society'),
    _t('History', 8,  'British East India Company',             'Arrival, trade, conquest, administrative policies'),
    _t('History', 8,  'Revolt of 1857',                         'Causes, events, consequences, significance'),
    _t('History', 8,  'Social Reform Movements',                'Raja Ram Mohan Roy, women rights, caste reform'),
    _t('History', 8,  'Indian National Movement',               'Congress, partition of Bengal, non-cooperation, civil disobedience'),
    _t('History', 9,  'French Revolution',                      'Causes, events, impact, liberty equality fraternity'),
    _t('History', 9,  'Rise of Nationalism in Europe',          'Romanticism, unification of Germany Italy, nation-states'),
    _t('History', 9,  'World War Causes and Effects',           'Alliances, imperialism, WWI WWII, consequences'),
    _t('History', 9,  'Russian Revolution',                     'Czarist Russia, Bolsheviks, socialism, USSR formation'),
    _t('History', 10, 'Nationalism in India',                   'Non-cooperation, civil disobedience, Quit India, partition'),
    _t('History', 10, 'Industrialisation',                      'Britain industrial revolution, factories, urbanisation'),
    _t('History', 10, 'Globalisation History',                  'Silk routes, trade, colonialism, modern globalisation'),
    _t('History', 11, 'Early Societies',                        'Paleolithic neolithic, agriculture revolution, bronze age'),
    _t('History', 11, 'Ancient Empires',                        'Mesopotamia Egypt Greece Rome, culture administration'),
    _t('History', 11, 'Three Orders Medieval Europe',           'Feudalism, church, crusades, black death'),
    _t('History', 12, 'Cold War',                               'USA USSR rivalry, arms race, proxy wars, détente'),
    _t('History', 12, 'Decolonisation',                         'African Asian independence movements, new nations'),
    _t('History', 12, 'Post Cold War World',                    'Unipolarity, globalisation, terrorism, new challenges'),

    # ── GEOGRAPHY Class 6-12 ─────────────────────────────────────

    _t('Geography', 6,  'Solar System and Earth',               'Planets, rotation revolution, seasons, solstice equinox'),
    _t('Geography', 6,  'Globe and Maps',                       'Latitudes longitudes, time zones, map types, scale'),
    _t('Geography', 6,  'Landforms',                            'Mountains plateaus plains, rivers, formation'),
    _t('Geography', 6,  'India Physical Features',              'Mountains plains desert coastline, rivers'),
    _t('Geography', 7,  'Inside the Earth',                     'Layers, rocks, volcanoes, earthquakes, plate tectonics'),
    _t('Geography', 7,  'Atmosphere',                           'Composition, layers, temperature, pressure, winds'),
    _t('Geography', 7,  'Water Bodies',                         'Ocean circulation, tides, coral reefs, sea resources'),
    _t('Geography', 7,  'Natural Vegetation and Wildlife',      'Factors, types of forests, grasslands, conservation'),
    _t('Geography', 7,  'Human Settlements',                    'Rural urban, types, functions, growth'),
    _t('Geography', 8,  'Resources',                            'Types, conservation, sustainable development'),
    _t('Geography', 8,  'Agriculture',                          'Types of farming, crops, green revolution, food security'),
    _t('Geography', 8,  'Industries',                           'Types, location factors, iron steel textile IT'),
    _t('Geography', 9,  'India Size and Location',              'Coordinates, neighbours, strategic importance'),
    _t('Geography', 9,  'Physical Features of India',           'Geological structure, mountains plains plateau coast'),
    _t('Geography', 9,  'Climate of India',                     'Factors, monsoon, seasons, rainfall distribution'),
    _t('Geography', 9,  'Population',                           'Size growth distribution density, composition, migration'),
    _t('Geography', 10, 'Resources and Development',            'Land soil degradation, conservation, land use pattern'),
    _t('Geography', 10, 'Water Resources',                      'Availability, dam multipurpose projects, conservation'),
    _t('Geography', 10, 'Agriculture in India',                 'Cropping pattern, land reforms, challenges, food security'),
    _t('Geography', 10, 'Minerals and Energy Resources',        'Types, distribution, conservation, energy crisis'),
    _t('Geography', 10, 'Manufacturing Industries',             'Importance, location, agro textile steel chemical auto'),
    _t('Geography', 10, 'Transport and Communication',          'Roadways railways waterways airways, trade, tourism'),
    _t('Geography', 11, 'Physical Geography Fundamentals',      'Geomorphology climatology oceanography biogeography'),
    _t('Geography', 12, 'Human Geography',                      'Nature scope, world population, migration, settlements'),
    _t('Geography', 12, 'Economic Geography',                   'Primary secondary tertiary activities, world trade'),

    # ── ECONOMICS Class 9-12 ─────────────────────────────────────

    _t('Economics', 9,  'Rural Economy',                        'Farming, land ownership, credit, infrastructure in villages'),
    _t('Economics', 9,  'People as Resource',                   'Human capital, education, health, unemployment'),
    _t('Economics', 9,  'Poverty',                              'Causes, poverty line, government measures, comparison'),
    _t('Economics', 9,  'Food Security',                        'Availability, access, absorption, buffer stock, PDS'),
    _t('Economics', 10, 'Development',                          'Different goals, income comparison, human development'),
    _t('Economics', 10, 'Sectors of Indian Economy',            'Primary secondary tertiary, organised unorganised, GDP'),
    _t('Economics', 10, 'Money and Credit',                     'Currency, banking, credit, formal informal, SHGs'),
    _t('Economics', 10, 'Globalisation',                        'MNCs, trade investment, impact on India, WTO'),
    _t('Economics', 10, 'Consumer Rights',                      'Consumer movement, rights, COPRA, redressal'),
    _t('Economics', 11, 'Introduction to Economics',            'Microeconomics macroeconomics, positive normative, scarcity'),
    _t('Economics', 11, 'Consumer Theory',                      'Utility, indifference curves, budget constraint, demand'),
    _t('Economics', 11, 'Production and Costs',                 'Production function, costs, short run long run'),
    _t('Economics', 11, 'Market Structures',                    'Perfect competition, monopoly, oligopoly, pricing'),
    _t('Economics', 12, 'National Income',                      'GDP GNP NNP, methods of measurement, circular flow'),
    _t('Economics', 12, 'Money and Banking',                    'Money supply, central bank, credit creation, monetary policy'),
    _t('Economics', 12, 'Income Determination',                 'Aggregate demand supply, multiplier, fiscal policy'),
    _t('Economics', 12, 'Indian Economy',                       'Planning, LPG reforms, poverty, agriculture industry services'),

    # ── COMPUTER SCIENCE Class 9-12 ──────────────────────────────

    _t('ComputerScience', 9,  'Computer Fundamentals',          'Hardware software, input output, memory, generations'),
    _t('ComputerScience', 9,  'Operating Systems',              'Functions, types, file management, GUI CLI'),
    _t('ComputerScience', 9,  'Internet Basics',                'Network, protocols, WWW, email, search engines, safety'),
    _t('ComputerScience', 10, 'Programming Concepts',           'Algorithm, flowchart, variables, control structures'),
    _t('ComputerScience', 10, 'Database Basics',                'DBMS, tables, relationships, basic SQL'),
    _t('ComputerScience', 10, 'Cybersecurity',                  'Threats, malware, passwords, safe practices'),
    _t('ComputerScience', 11, 'Python Fundamentals',            'Syntax, data types, operators, input output'),
    _t('ComputerScience', 11, 'Control Flow in Python',         'Conditionals loops, functions, recursion'),
    _t('ComputerScience', 11, 'Data Structures',                'Lists tuples sets dicts, stacks queues, searching sorting'),
    _t('ComputerScience', 11, 'Computer Networks',              'OSI model, protocols, IP addressing, internet'),
    _t('ComputerScience', 11, 'Database and SQL',               'Relational model, DDL DML, joins, normalization'),
    _t('ComputerScience', 12, 'Object Oriented Programming',    'Classes objects, inheritance polymorphism, encapsulation'),
    _t('ComputerScience', 12, 'Advanced Data Structures',       'Trees graphs, hashing, algorithms, complexity'),
    _t('ComputerScience', 12, 'Web Development Basics',         'HTML CSS, HTTP, web servers, client server model'),
    _t('ComputerScience', 12, 'Artificial Intelligence Basics', 'ML concepts, supervised unsupervised, applications'),
]


def get_all_topics() -> List[Topic]:
    return SYLLABUS


def get_topics_for_grade(grade: int) -> List[Topic]:
    return [t for t in SYLLABUS if t.grade == grade]


def get_topics_for_subject(subject: str) -> List[Topic]:
    return [t for t in SYLLABUS if t.subject == subject]


def total_topics() -> int:
    return len(SYLLABUS)


if __name__ == '__main__':
    print(f"Total topics in syllabus: {total_topics()}")
    subjects = {}
    for t in SYLLABUS:
        subjects[t.subject] = subjects.get(t.subject, 0) + 1
    for s, c in sorted(subjects.items()):
        print(f"  {s:<20} {c} topics")