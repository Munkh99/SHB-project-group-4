from collections import defaultdict

sensors_config = {
    'applicationevent': defaultdict(list,
                                    # keep the following categories: app_communication, app_education, app_maps_and_navigation,
                                    # app_social, app_travel_and_local, app_productivity
                                    columns_to_exclude=['experiment_id'],
                                    normalization=['app_nunique', 'app_category_nunique']),

    'locationeventpertime': defaultdict(list),
    'timediary': defaultdict(list,
                             columns_to_exclude=['YY', 'MM', 'DD', 'hh', 'mm', ]),
    'notificationevent': defaultdict(list,
                                     normalization=['notification_posted', 'notification_removed']),
}

application_grouping = {
    'productivity_and_admin':
        [
            'productivity',
            'tools',
            'education',
            'art_and_design',
            'business',
            'books_and_reference',
            'finance',
            'personalization',
            'comics',
            'libraries_and_demo'
        ],
    'health':
        [
            'lifestyle',
            'health_and_fitness',
            'parenting',
            'dating',
            'medical',
            'beauty',
            'house_and_home',
            'sports',

        ],
    'internet_and_social_media':
        [
            'communication',
            'entertainment',
            'news_and_magazines',
            'social'],
    'maps_and_travel':
        [
            'maps_and_navigation',
            'auto_and_vehicles',
            'travel_and_local',
            'weather'
        ],
    'media':
        [
            'music_and_audio',
            'photography',
            'video_players'],
    'shopping':
        [
            'shopping',
        ],
    'food_and_drink':
        ['food_and_drink'],
    'games':
        [
            'game_action',
            'game_adventure',
            'game_arcade',
            'game_board',
            'game_card',
            'game_casual',
            'game_card',
            'game_educational',
            'game_music',
            'game_puzzle',
            'game_racing',
            'game_role_playing',
            'game_simulation',
            'game_strategy',
            'game_sports',
            'game_trivia',
            'game_word',
        ],
    'other': [
        'not_found'
    ]
}

app_not_found_suffix = {
    'productivity_and_admin':
        [
            'settings', 'vending', 'launcher', 'home', 'deskclock', 'notepad', 'note', 'contacts', 'reminder',
            'dictionary', 'calculator', 'safe', 'security', 'vpn', 'syste', 'wallpaper', 'plugin', 'instantshare',
            'wallet', 'batterysaver', 'settings', 'alarmclock', 'installer', 'htmlviewer', 'reader',
            'book', 'clean', 'theme', 'permissioncontroller', 'smartshot', 'capture',
            'service', 'documents', 'scan', 'appcloner', 'screenshot', 'compass', 'calendar', 'share',
            'imagecolorizerpro', 'folder', 'galaxyfinder', 'printspooler', 'module', 'antivirus', 'wifi',
            'storage', 'disk', 'manga', 'bank', 'coloros', 'management', 'assistant', 'book', 'learn', 'manager',
            'auth', 'bluetooth', 'sharing', 'notification',  'study', 'organizer', 'mirroring', 'guard',
            'billing', 'battery', 'sketch', 'tasker', 'share', 'update', 'roommate', 'equalizer', 'applocker',
            'pdfviewer', 'power', 'quiz', 'lock','shortcut', 'draw', 'remotecontroller', 'sticker', 'clock', 'record',
            'setting', 'snappage', 'carige', 'office', 'accessibility',
            'it.raiffeisen.mob', 'eu.kanade.tachiyomi'
            'it.asmset.app.gaseluce',
            'ch.bitspin.timely',
            'org.lineageos.lineageparts',
            'il.yavji.volumecontrolads'


        ],
    'health':
        [
            'health', 'life', 'parent', 'routine', 'wellbeing', 'sport', 'biometrics', 'yoga'
            'com.vratsev.bpexpro', 'leone.moredots.rightapp', 'com.gotokeep.yoga.intl'
        ],
    'internet_and_social_media':
        [
            'dialer', 'facebook', 'instagram', 'talk', 'youtube', 'browser', 'movie', 'email',
            'telegram', 'messenger', 'mozilla', 'insta', 'follow', 'tube', 'messaging', 'signin', 'telecom', 'phone',
            'search', 'hashtags', 'conversations', 'social',
            'it.quadronica.fantagazzetta',
            'com.mobilefootie.fotmobpro',
        ],
    'maps_and_travel':
        [
            'weather', 'trains', 'smartcommunitylab', 'meteotrentino','bus',
            'ru.gismeteo.gismeteo', 'air.lovby',
        ],
    'media':
        [
            'camera', 'gallery', 'himovie', 'music', 'video', 'photo', 'time.lapse', 'player', 'homedj', 'slowmotion',
            'retouch', 'sound', 'hub', 'radio', 'tv', 'media', 'image', 'spotify', 'snapcam','appmarket',
            'com.digitalmosaik.trentinovr', 'com.naver.vapp', 'fast.motion.app'
            'aremoji', 'arzone',
        ],
    'shopping':
        [
            'ikea', 'store', 'ovs', 'shop', 'amazon', 'dressroom', 'samsungapps',
            'com.auto.usate.in.italia', 'it.pinalli.app', 'it.softecspa.icoop.main'
        ],
    'food_and_drink':
        [
            'food','pizza', 'umami', 'nogood', 'it.mulinobianco.miomulino',
        ],
    'games':

        [
            'game', 'play', 'offroad', 'apex', 'netmarble',
            'com.ubisoft.redlynx.trialsfrontier.ggp', 'klb.android.lovelive_en'
        ],
    'other': [
        'android',
    ]

}
