import pprint
from wordcel.rag.contextual_retrieval import ContextualRetrieval

capybara = """
The capybara[a] or greater capybara (Hydrochoerus hydrochaeris) is the largest living rodent[2], native to South America. It is a member of the genus Hydrochoerus. The only other extant member is the lesser capybara (Hydrochoerus isthmius). Its close relatives include guinea pigs and rock cavies, and it is more distantly related to the agouti, the chinchilla, and the nutria. The capybara inhabits savannas and dense forests, and lives near bodies of water. It is a highly social species and can be found in groups as large as 100 individuals, but usually live in groups of 10–20 individuals. The capybara is hunted for its meat and hide and also for grease from its thick fatty skin.[3]

Etymology
Its common name is derived from Tupi ka'apiûara, a complex agglutination of kaá (leaf) + píi (slender) + ú (eat) + ara (a suffix for agent nouns), meaning "one who eats slender leaves", or "grass-eater".[4] The genus name, hydrochoerus, comes from Greek ὕδωρ (hydor "water") and χοῖρος (choiros "pig, hog") and the species name, hydrochaeris, comes from Greek ὕδωρ (hydor "water") and χαίρω (chairo "feel happy, enjoy").[5][6]

Classification and phylogeny
The capybara and the lesser capybara both belong to the subfamily Hydrochoerinae along with the rock cavies. The living capybaras and their extinct relatives were previously classified in their own family Hydrochoeridae.[7] Since 2002, molecular phylogenetic studies have recognized a close relationship between Hydrochoerus and Kerodon, the rock cavies,[8] supporting placement of both genera in a subfamily of Caviidae.[5]

Paleontological classifications previously used Hydrochoeridae for all capybaras, while using Hydrochoerinae for the living genus and its closest fossil relatives, such as Neochoerus,[9][10] but more recently have adopted the classification of Hydrochoerinae within Caviidae.[11] The taxonomy of fossil hydrochoerines is also in a state of flux. In recent years, the diversity of fossil hydrochoerines has been substantially reduced.[9][10] This is largely due to the recognition that capybara molar teeth show strong variation in shape over the life of an individual. In one instance, material once referred to four genera and seven species on the basis of differences in molar shape is now thought to represent differently aged individuals of a single species, Cardiatherium paranense.[9] Among fossil species, the name "capybara" can refer to the many species of Hydrochoerinae that are more closely related to the modern Hydrochoerus than to the "cardiomyine" rodents like Cardiomys. The fossil genera Cardiatherium, Phugatherium, Hydrochoeropsis, and Neochoerus are all capybaras under that concept.[11]

Description

Taxidermy specimen of a capybara
The capybara has a heavy, barrel-shaped body and short head, with reddish-brown fur on the upper part of its body that turns yellowish-brown underneath. Its sweat glands can be found in the surface of the hairy portions of its skin, an unusual trait among rodents.[7] The animal lacks down hair, and its guard hair differs little from over hair.[12]


Capybara skeleton
Adult capybaras grow to 106 to 134 cm (3.48 to 4.40 ft) in length, stand 50 to 62 cm (20 to 24 in) tall at the withers, and typically weigh 35 to 66 kg (77 to 146 lb), with an average in the Venezuelan llanos of 48.9 kg (108 lb).[13][14][15] Females are slightly heavier than males. The top recorded weights are 91 kg (201 lb) for a wild female from Brazil and 73.5 kg (162 lb) for a wild male from Uruguay.[7][16] Also, an 81 kg individual was reported in São Paulo in 2001 or 2002.[17] The dental formula is 
1.0.1.3
1.0.1.3
. Capybaras have slightly webbed feet and vestigial tails.[7] Their hind legs are slightly longer than their forelegs; they have three toes on their rear feet and four toes on their front feet.[18] Their muzzles are blunt, with nostrils, and the eyes and ears are near the top of their heads.

Its karyotype has 2n = 66 and FN = 102, meaning it has 66 chromosomes with a total of 102 arms.[5][7]

Ecology

Yellow-headed caracara sitting on a capybara

A family of capybara swimming
Capybaras are semiaquatic mammals[15] found throughout all countries of South America except Chile.[19] They live in densely forested areas near bodies of water, such as lakes, rivers, swamps, ponds, and marshes,[14] as well as flooded savannah and along rivers in the tropical rainforest. They are superb swimmers and can hold their breath underwater for up to five minutes at a time. Capybara have flourished in cattle ranches. They roam in home ranges averaging 10 hectares (25 acres) in high-density populations.[7]

Many escapees from captivity can also be found in similar watery habitats around the world. Sightings are fairly common in Florida, although a breeding population has not yet been confirmed.[20] In 2011, one specimen was spotted on the Central Coast of California.[21] These escaped populations occur in areas where prehistoric capybaras inhabited; late Pleistocene capybaras inhabited Florida[22] and Hydrochoerus hesperotiganites in California and Hydrochoerus gaylordi in Grenada, and feral capybaras in North America may actually fill the ecological niche of the Pleistocene species.[23]
"""

kerodon = """
The genus Kerodon (vernacular name mocos; rock cavies[1]) contains two species of South American rock cavies, related to capybaras and guinea pigs.[2] They are found in semiarid regions of northeast Brazil known as the Caatinga. This area has a rocky terrain with large granite boulders that contain rifts and hollows where Kerodon species primarily live.[3]

Characteristics
They are hystricomorph rodents, medium-sized, with rabbit-like bodies, a squirrel-like face, and heavily padded feet. Their nails are blunt on all digits except a small grooming claw on the outermost digit of the foot. Fully grown adults weigh around 1000 g or 31-35 oz, and range in length from 200 to 400 mm or 7.5 to 16 in.[4] They forage for mostly leaves, grasses, seeds, and tree bark.[3] They breed year round, usually having one to three litters per year and one to three young per pregnancy. Gestation last around 76 days and the young are weaned from the mother within 33 days. They reach sexual maturity at 133 days.[citation needed]

Behavior
Like their relatives, the capybaras and the maras, members of the genus Kerodon are highly social.[5] Kerodon species, like capybaras, are polygynous, with males forming harems. They are very vocal creatures and produce various whistles, chirps, and squeaks.[4] Males establish ownership over one or several rock piles and defend their territories. Within each group, a hierarchical structure exists. They are primarily active during late hours of the day.[citation needed]

Classification
Traditionally, the genus Kerodon has been considered a member of the subfamily Caviinae along with the guinea pigs and other cavies. Molecular results have consistently suggested Kerodon is most closely related to the capybara, and the two evolved from within the Caviidae.[5] This led Woods and Kilpatrick (2005) to unite the two into the subfamily Hydrochoerinae within the Caviidae. Using a molecular clock approach, Opazo[6] suggested Kerodon diverged from Hydrochoerus (the capybara) in the late Middle Miocene.
"""


nate_silver = """
Nathaniel Read Silver (born January 13, 1978) is an American statistician, writer, and poker player who analyzes baseball, basketball, and elections. He is the founder of FiveThirtyEight, and held the position of editor-in-chief there, along with being a special correspondent for ABC News, until May 2023.[2] Since departing FiveThirtyEight, Silver has been publishing on his Substack blog Silver Bulletin[3] and serves as an advisor to Polymarket.[4]
 
Silver was named one of the world's 100 most influential people by Time in 2009 after an election forecasting system he developed successfully predicted the outcomes in forty-nine of the fifty states in the 2008 U.S. presidential election.[5] His subsequent election forecasting systems predicted the outcome of the 2012 and 2020 presidential elections with a high degree of accuracy. His polls-only model gave Donald Trump, the ultimate winner, only a 28.6% chance of victory in the 2016 presidential election,[6] although this was higher than any other forecasting competitors.[7]

Much of Silver's approach can be characterized by using probabilistic and statistical modeling to try to understand complex social systems, such as professional sports, the popularity of political platforms, and elections.
"""


docs = [capybara, kerodon, nate_silver] 

def test_indexing_and_save():
    retriever = ContextualRetrieval(docs)
    retriever.index_documents()
    print(retriever.retrieve("what genus is a capybara?"))
    retriever.save("retriever.pkl")

def test_load_and_retrieve():
    retriever = ContextualRetrieval.from_saved("retriever.pkl")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(retriever.retrieve("what genus is a capybara?"))
    print(retriever.generate("what genus is a capybara?"))

test_indexing_and_save()
test_load_and_retrieve()