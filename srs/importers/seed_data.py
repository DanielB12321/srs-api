"""
Static reference data for the importer.

Populates the Element table on first run. Units follow OSNACA convention:
  - g/t for precious metals (Au, Ag, PGEs)
  - %   for major rock-forming elements
  - ppm for everything else
"""

GRAMS_PER_TONNE = "g/t"
PERCENT = "%"
PPM = "ppm"

ELEMENTS = [
    # symbol, name, atomic_number, default_unit
    ("Ag",  "Silver",       47, GRAMS_PER_TONNE),
    ("Al",  "Aluminium",    13, PERCENT),
    ("As",  "Arsenic",      33, PPM),
    ("Au",  "Gold",         79, GRAMS_PER_TONNE),
    ("B",   "Boron",         5, PPM),
    ("Ba",  "Barium",       56, PPM),
    ("Be",  "Beryllium",     4, PPM),
    ("Bi",  "Bismuth",      83, PPM),
    ("Ca",  "Calcium",      20, PERCENT),
    ("Cd",  "Cadmium",      48, PPM),
    ("Ce",  "Cerium",       58, PPM),
    ("Cl",  "Chlorine",     17, PPM),
    ("Co",  "Cobalt",       27, PPM),
    ("Cr",  "Chromium",     24, PPM),
    ("Cs",  "Caesium",      55, PPM),
    ("Cu",  "Copper",       29, PPM),
    ("Dy",  "Dysprosium",   66, PPM),
    ("Er",  "Erbium",       68, PPM),
    ("Eu",  "Europium",     63, PPM),
    ("Fe",  "Iron",         26, PERCENT),
    ("Ga",  "Gallium",      31, PPM),
    ("Gd",  "Gadolinium",   64, PPM),
    ("Ge",  "Germanium",    32, PPM),
    ("Hf",  "Hafnium",      72, PPM),
    ("Hg",  "Mercury",      80, PPM),
    ("Ho",  "Holmium",      67, PPM),
    ("In",  "Indium",       49, PPM),
    ("Ir",  "Iridium",      77, GRAMS_PER_TONNE),
    ("K",   "Potassium",    19, PERCENT),
    ("La",  "Lanthanum",    57, PPM),
    ("Li",  "Lithium",       3, PPM),
    ("Lu",  "Lutetium",     71, PPM),
    ("Mg",  "Magnesium",    12, PERCENT),
    ("Mn",  "Manganese",    25, PERCENT),
    ("Mo",  "Molybdenum",   42, PPM),
    ("Na",  "Sodium",       11, PERCENT),
    ("Nb",  "Niobium",      41, PPM),
    ("Nd",  "Neodymium",    60, PPM),
    ("Ni",  "Nickel",       28, PPM),
    ("Os",  "Osmium",       76, GRAMS_PER_TONNE),
    ("P",   "Phosphorus",   15, PERCENT),
    ("Pb",  "Lead",         82, PPM),
    ("Pd",  "Palladium",    46, GRAMS_PER_TONNE),
    ("Pr",  "Praseodymium", 59, PPM),
    ("Pt",  "Platinum",     78, GRAMS_PER_TONNE),
    ("Rb",  "Rubidium",     37, PPM),
    ("Re",  "Rhenium",      75, PPM),
    ("Rh",  "Rhodium",      45, GRAMS_PER_TONNE),
    ("Ru",  "Ruthenium",    44, GRAMS_PER_TONNE),
    ("S",   "Sulfur",       16, PERCENT),
    ("Sb",  "Antimony",     51, PPM),
    ("Sc",  "Scandium",     21, PPM),
    ("Se",  "Selenium",     34, PPM),
    ("Si",  "Silicon",      14, PERCENT),
    ("Sm",  "Samarium",     62, PPM),
    ("Sn",  "Tin",          50, PPM),
    ("Sr",  "Strontium",    38, PPM),
    ("Ta",  "Tantalum",     73, PPM),
    ("Tb",  "Terbium",      65, PPM),
    ("Te",  "Tellurium",    52, PPM),
    ("Th",  "Thorium",      90, PPM),
    ("Ti",  "Titanium",     22, PERCENT),
    ("Tl",  "Thallium",     81, PPM),
    ("Tm",  "Thulium",      69, PPM),
    ("U",   "Uranium",      92, PPM),
    ("V",   "Vanadium",     23, PPM),
    ("W",   "Tungsten",     74, PPM),
    ("Y",   "Yttrium",      39, PPM),
    ("Yb",  "Ytterbium",    70, PPM),
    ("Zn",  "Zinc",          30, PPM),
    ("Zr",  "Zirconium",    40, PPM),
]


# Columns in the Data sheet that are NOT element measurements
NON_ELEMENT_COLUMNS = {"Sample", "Code", "Code Tester", "LOI", "Wt Tot"}


# Maps composite headers (row1 + row2) to (element_symbol, analytical_method)
# Used by the importer to split "Au (FA)" vs "Au (AR)" into two methods on Au.
ELEMENT_METHOD_OVERRIDES = {
    ("AU1", "Au (FA)"): ("Au", "FA"),
    ("Au AR", "Au (AR)"): ("Au", "AR"),
}
