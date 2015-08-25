import requests
from bs4 import BeautifulSoup as bs

r = requests.get("https://en.wikipedia.org/wiki/Grumpy_Cat")
r.status_code
soup = bs(r.text, 'html.parser')
anchors = soup.findAll('a', {'class': ['show-visited']})

url = 'https://macaulaylibrary.org/audio/'
species = ['ML_American_Crow_2015_Aug_22_14_17_38.csv',
           'ML_American_Goldfinch_2015_Aug_22_14_13_06.csv',
           'ML_American_Robin_2015_Aug_22_14_04_24.csv',
           'ML_Annas_Hummingbird_2015_Aug_22_14_26_02.csv',
           'ML_Bald_Eagle_2015_Aug_22_14_30_15.csv',
           'ML_Barn_Swallow_2015_Aug_22_14_08_46.csv',
           'ML_Barred_Owl_2015_Aug_22_14_35_45.csv',
           'ML_Belted_Kingfisher_2015_Aug_22_14_33_00.csv',
           'ML_Bewicks_Wren_2015_Aug_22_13_58_58.csv',
           'ML_Black-capped_Chickadee_2015_Aug_22_14_01_33.csv',
           'ML_Black-headed_Grosbeak_2015_Aug_22_14_14_09.csv',
           'ML_Black-throated_Gray_Warbler_2015_Aug_22_14_25_19.csv',
           'ML_Brewers_Blackbird_2015_Aug_22_14_34_40.csv',
           'ML_Brown-headed_Cowbird_2015_Aug_22_14_28_46.csv',
           'ML_Brown_Creeper_2015_Aug_22_14_15_33.csv',
           'ML_Bushtit_2015_Aug_22_14_16_13.csv',
           'ML_Cackling_Goose_2015_Aug_22_14_37_20.csv',
           'ML_Canada_Goose_2015_Aug_22_14_37_44.csv',
           'ML_Cedar_Waxwing_2015_Aug_23_21_19_54.csv',
           'ML_Chestnut-backed_Chickadee_2015_Aug_22_14_02_12.csv',
           'ML_Common_Raven_2015_Aug_23_21_50_44.csv',
           'ML_Coopers_Hawk_2015_Aug_22_14_31_19.csv',
           'ML_Dark-eyed_Junco_2015_Aug_22_14_06_48.csv',
           'ML_Downy_Woodpecker_2015_Aug_22_14_21_28.csv',
           'ML_European_Starling_2015_Aug_22_14_29_22.csv',
           'ML_Evening_Grosbeak_2015_Aug_22_14_14_37.csv',
           'ML_Fox_Sparrow_2015_Aug_22_14_06_19.csv',
           'ML_Glaucous-winged_Gull_2015_Aug_22_14_19_27.csv',
           'ML_Golden-crowned_Kinglet_2015_Aug_22_14_02_39.csv',
           'ML_Golden-crowned_Sparrow_2015_Aug_22_14_07_57.csv',
           'ML_Hairy_Woodpecker_2015_Aug_22_14_21_49.csv',
           'ML_Hermit_Thrush_2015_Aug_22_14_05_16.csv',
           'ML_House_Finch_2015_Aug_22_14_11_33.csv',
           'ML_House_Sparrow_2015_Aug_22_14_29_52.csv',
           'ML_Mew_Gull_2015_Aug_22_14_20_11.csv',
           'ML_Northern_Flicker_2015_Aug_22_14_21_05.csv',
           'ML_Northwestern_Crow_2015_Aug_22_14_18_04.csv',
           'ML_Orange-crowned_Warbler_2015_Aug_22_14_24_49.csv',
           'ML_Osprey_2015_Aug_22_14_30_49.csv',
           'ML_Pacific-slope_Flycatcher_2015_Aug_22_14_17_11.csv',
           'ML_Pacific_Wren_2015_Aug_22_14_03_28.csv',
           'ML_Pileated_Woodpecker_2015_Aug_22_14_22_41.csv',
           'ML_Pine_Siskin_2015_Aug_22_14_13_34.csv',
           'ML_Purple_Finch_2015_Aug_22_14_12_10.csv',
           'ML_Red-breasted_Nuthatch_2015_Aug_22_14_15_05.csv',
           'ML_Red-breasted_Sapsucker_2015_Aug_22_14_22_15.csv',
           'ML_Red-tailed_Hawk_2015_Aug_22_14_32_31.csv',
           'ML_Red-winged_Blackbird_2015_Aug_22_14_33_31.csv',
           'ML_Red_Crossbill_2015_Aug_22_14_10_33.csv',
           'ML_Ring-billed_Gull_2015_Aug_22_14_20_36.csv',
           'ML_Ruby-crowned_Kinglet_2015_Aug_22_14_03_03.csv',
           'ML_Rufous_Hummingbird_2015_Aug_22_14_26_48.csv',
           'ML_Rusty_Blackbird_2015_Aug_22_14_35_18.csv',
           'ML_Sharp-shinned_Hawk_2015_Aug_22_14_31_46.csv',
           'ML_Song_Sparrow_2015_Aug_22_13_53_38.csv',
           'ML_Spotted_Owl_Northern_2015_Aug_22_14_36_28.csv',
           'ML_Spotted_Towhee_2015_Aug_23_21_19_26.csv',
           'ML_Stellers_Jay_2015_Aug_22_14_27_38.csv',
           'ML_Swainsons_Thrush_2015_Aug_22_14_04_50.csv',
           'ML_Townsends_Warbler_2015_Aug_22_14_23_24.csv',
           'ML_Tree_Swallow_2015_Aug_22_14_09_06.csv',
           'ML_Varied_Thrush_2015_Aug_22_14_05_41.csv',
           'ML_Violet-green_Swallow_2015_Aug_22_14_09_35.csv',
           'ML_Warbling_Vireo_2015_Aug_22_14_46_42.csv',
           'ML_Western_Gull_2015_Aug_22_14_19_50.csv',
           'ML_Western_Scrub-Jay_2015_Aug_22_14_28_06.csv',
           'ML_Western_Tanager_2015_Aug_22_14_16_44.csv',
           'ML_White-crowned_Sparrow_2015_Aug_22_14_07_32.csv',
           'ML_Wilsons_Warbler_2015_Aug_22_14_24_15.csv',
           'ML_Winter_Wren_2015_Aug_22_14_03_52.csv',
           'ML_Yellow-rumped_Warbler_2015_Aug_22_14_23_47.csv']

codes = ['AMCR', 'AMGO', 'AMRO', 'ANHU', 'BAEA', 'BARS', 'BAOW', 'BEKI',
         'BEWR', 'BCCH', 'BHGR', 'BTYW', 'BRBL', 'BHCO', 'BRCR', 'BUSH',
         'CACG', 'CAGO', 'CEWA', 'CORA', 'CBCH', 'COHA', 'DEJU', 'DOWO',
         'EUST', 'EVGR', 'FOSP', 'GWGU', 'GCKI', 'GCSP', 'HAWO', 'HETH',
         'HOFI', 'HOSP', 'MEGU', 'NOFL', 'NOCR', 'OCWA', 'OSPR', 'PSFL',
         'PAWR', 'PIWO', 'PISI', 'PUFI', 'RBNU', 'RBSA', 'RTHA', 'RWBL',
         'RECR', 'RBGU', 'RCKI', 'RUHU', 'RUBL', 'SSHA', 'SOSP', 'SPOW',
         'SPTO', 'STJA', 'SWTH', 'TOWA', 'TRES', 'VATH', 'VGSW', 'WAVI',
         'WEGU', 'WESJ', 'WETA', 'WCSP', 'WIWA', 'WIWR', 'YRWA']

common_names = ['American Crow',
                'American Goldfinch',
                'American Robin',
                'Annas Hummingbird',
                'Bald Eagle',
                'Barn Swallow',
                'Barred Owl',
                'Belted Kingfisher',
                'Bewicks Wren',
                'Black-capped Chickadee',
                'Black-headed Grosbeak',
                'Black-throated Gray Warbler',
                'Brewers Blackbird',
                'Brown-headed Cowbird',
                'Brown Creeper',
                'Bushtit',
                'Cackling Goose',
                'Canada Goose',
                'Cedar Waxwing',
                'Chestnut-backed Chickadee',
                'Common Raven',
                'Coopers Hawk',
                'Dark-eyed Junco',
                'Downy Woodpecker',
                'European Starling',
                'Evening Grosbeak',
                'Fox Sparrow',
                'Glaucous-winged Gull',
                'Golden-crowned Kinglet',
                'Golden-crowned Sparrow',
                'Hairy Woodpecker',
                'Hermit Thrush',
                'House Finch',
                'House Sparrow',
                'Mew Gull',
                'Northern Flicker',
                'Northwestern Crow',
                'Orange-crowned Warbler',
                'Osprey',
                'Pacific-slope Flycatcher',
                'Pacific Wren',
                'Pileated Woodpecker',
                'Pine Siskin',
                'Purple Finch',
                'Red-breasted Nuthatch',
                'Red-breasted Sapsucker',
                'Red-tailed Hawk',
                'Red-winged Blackbird',
                'Red Crossbill',
                'Ring-billed Gull',
                'Ruby-crowned Kinglet',
                'Rufous Hummingbird',
                'Rusty Blackbird',
                'Sharp-shinned Hawk',
                'Song Sparrow',
                'Spotted Owl',
                'Spotted Towhee',
                'Stellers Jay',
                'Swainsons Thrush',
                'Townsends Warbler',
                'Tree Swallow',
                'Varied Thrush',
                'Violet-green Swallow',
                'Warbling Vireo',
                'Western Gull',
                'Western Scrub-Jay',
                'Western Tanager',
                'White-crowned Sparrow',
                'Wilsons Warbler',
                'Winter Wren',
                'Yellow-rumped Warbler']
# try:
#     # Python 3.x
#     from urllib.request import urlopen, urlretrieve, quote
#     from urllib.parse import urljoin
# except ImportError:
#     # Python 2.x
#     from urllib import urlopen, urlretrieve, quote
#     from urlparse import urljoin
#
# from bs4 import BeautifulSoup
#
# url = 'http://oilandgas.ky.gov/Pages/ProductionReports.aspx'
# u = urlopen(url)
# try:
#     html = u.read().decode('utf-8')
# finally:
#     u.close()
#
# soup = BeautifulSoup(html)
# for link in soup.select('div[webpartid] a'):
#     href = link.get('href')
#     if href.startswith('javascript:'):
#         continue
#     filename = href.rsplit('/', 1)[-1]
#     href = urljoin(url, quote(href))
#     try:
#         urlretrieve(href, filename)
#     except:
#         print('failed to download')