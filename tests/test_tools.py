import unittest, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from hearing import tools


class AudiogramTests(unittest.TestCase):
    NORMAL = {250: 10, 500: 10, 1000: 10, 2000: 10, 4000: 10, 8000: 10}
    SLOPING = {250: 15, 500: 20, 1000: 30, 2000: 50, 4000: 65, 8000: 75}
    FLAT_MOD = {250: 50, 500: 50, 1000: 50, 2000: 50, 4000: 50, 8000: 50}

    def test_normal_hearing_hears_everything(self):
        a = tools.analyze_audiogram(self.NORMAL, self.NORMAL)
        self.assertEqual(a["ears"]["right"]["degree"], "normal")
        self.assertEqual(a["missed_sounds"], [])

    def test_sloping_loses_high_consonants_not_vowels(self):
        a = tools.analyze_audiogram(self.SLOPING, self.SLOPING)
        self.assertIn("s", a["missed_sounds"])
        self.assertIn("th", a["missed_sounds"])
        self.assertNotIn("oo", a["missed_sounds"])
        self.assertTrue(a["high_frequency_trouble"])
        self.assertIn("sloping", a["ears"]["left"]["configuration"])

    def test_flat_moderate(self):
        a = tools.analyze_audiogram(self.FLAT_MOD, None)
        self.assertEqual(a["ears"]["right"]["degree"], "moderate")
        self.assertNotIn("left", a["ears"])
        self.assertIn("flat", a["ears"]["right"]["configuration"])

    def test_better_ear_wins(self):
        a = tools.analyze_audiogram(self.SLOPING, self.NORMAL)
        self.assertEqual(a["missed_sounds"], [])
        self.assertIsNotNone(a["asymmetry"])

    def test_rejects_bad_input(self):
        with self.assertRaises(tools.AudiogramError):
            tools.analyze_audiogram({250: 10, 500: 10}, None)  # too few points
        with self.assertRaises(tools.AudiogramError):
            tools.analyze_audiogram({**self.NORMAL, 1000: 500}, None)
        with self.assertRaises(tools.AudiogramError):
            tools.analyze_audiogram({**self.NORMAL, "abc": 5}, None)
        with self.assertRaises(tools.AudiogramError):
            tools.analyze_audiogram({123: 5, 500: 5, 1000: 5}, None)

    def test_mishear_examples_follow_missed_sounds(self):
        a = tools.analyze_audiogram(self.SLOPING, self.SLOPING)
        self.assertTrue(any("fifty" in e for e in a["mishear_examples"]) or a["mishear_examples"])

    def test_degree_boundaries(self):
        self.assertEqual(tools.degree(25), "normal")
        self.assertEqual(tools.degree(26), "mild")
        self.assertEqual(tools.degree(91), "profound")


class AcousticsTests(unittest.TestCase):
    def test_loud_room_scores_hard(self):
        r = tools.score_acoustics(["hard_surfaces", "background_music", "crowded", "open_kitchen"])
        self.assertGreaterEqual(r["score"], 7)
        self.assertTrue(r["verdict"].startswith("Hard"))

    def test_quiet_room_scores_easy(self):
        r = tools.score_acoustics(["quiet_room", "small_group", "captioning_available"])
        self.assertEqual(r["score"], 1.0)

    def test_profile_changes_score(self):
        f = ["hard_surfaces", "dim_lighting"]
        base = tools.score_acoustics(f)["score"]
        worse = tools.score_acoustics(f, high_freq_trouble=True, relies_on_lipreading=True)["score"]
        self.assertGreater(worse, base)

    def test_unknown_and_duplicate_features(self):
        r = tools.score_acoustics(["hard_surfaces", "hard_surfaces", "made_up"])
        self.assertEqual(len(r["contributions"]), 1)
        self.assertEqual(r["ignored_unknown_features"], ["made_up"])

    def test_score_is_clamped(self):
        r = tools.score_acoustics(list(f for f, v in tools.FEATURES.items() if v[0] > 0) * 3)
        self.assertLessEqual(r["score"], 10)


class CallTests(unittest.TestCase):
    def test_known_and_unknown_org(self):
        self.assertEqual(tools.find_call_alternatives("medical")["organization_type"], "medical")
        self.assertEqual(tools.find_call_alternatives("zzz")["organization_type"], "other")
        self.assertTrue(tools.find_call_alternatives("bank_or_financial")["phone_relay_services"])


class LossTypeTests(unittest.TestCase):
    """A real conductive audiogram (bilateral atresia, 2015): flat air ~65 dB, bone near normal."""
    AIR_R = {250: 60, 500: 65, 1000: 65, 2000: 70, 4000: 65, 8000: 55}
    AIR_L = {250: 70, 500: 65, 1000: 65, 2000: 65, 4000: 70, 8000: 60}
    BONE_R = {250: 5, 500: 5, 1000: 5, 2000: 10, 4000: 5}
    BONE_L = {250: 10, 500: 15, 1000: 15, 2000: 20, 4000: 15}

    def test_conductive_is_detected(self):
        a = tools.analyze_audiogram(self.AIR_R, self.AIR_L, self.BONE_R, self.BONE_L)
        for ear in ("right", "left"):
            self.assertEqual(a["ears"][ear]["loss_type"], "conductive")
            self.assertGreaterEqual(a["ears"][ear]["air_bone_gap_db"], 50)
        self.assertTrue(a["loss_type_notes"][0].startswith("Sound is blocked"))

    def test_sensorineural_and_mixed(self):
        air = {500: 50, 1000: 55, 2000: 60, 4000: 70}
        self.assertEqual(tools.analyze_audiogram(air, None, {500: 45, 1000: 50, 2000: 55, 4000: 65})
                         ["ears"]["right"]["loss_type"], "sensorineural")
        self.assertEqual(tools.analyze_audiogram(air, None, {500: 30, 1000: 35, 2000: 40, 4000: 45})
                         ["ears"]["right"]["loss_type"], "mixed")

    def test_no_bone_numbers_means_unknown(self):
        a = tools.analyze_audiogram(self.AIR_R, self.AIR_L)
        self.assertTrue(a["ears"]["right"]["loss_type"].startswith("unknown"))
        self.assertEqual(a["loss_type_notes"], [])

    def test_bone_at_8k_is_rejected(self):
        with self.assertRaises(tools.AudiogramError):
            tools.analyze_audiogram(self.AIR_R, None, {8000: 10})

    def test_one_sided_implant_sets_seat_side_and_notes(self):
        a = tools.analyze_audiogram(self.AIR_R, self.AIR_L, self.BONE_R, self.BONE_L,
                                    "bone_conduction_implant", "left")
        self.assertEqual(a["seat_side"], "left")
        self.assertTrue(a["single_sided"])
        notes = " ".join(a["device"]["notes"])
        self.assertIn("clear, not just louder", notes)  # conductive ear + bone conduction
        self.assertIn("sit or walk on the left", notes)
        self.assertIn("microphone is on the left", notes)
        self.assertIn("reaches both inner ears", notes)
        self.assertIn("bone-conduction implant on the left side", tools.profile_summary(a))

    def test_bad_device_is_ignored(self):
        a = tools.analyze_audiogram(self.AIR_R, None, device="laser_ears", device_side="up")
        self.assertEqual(a["device"]["type"], "none")
        self.assertIsNone(a["seat_side"])

    def test_one_sided_hearing_makes_noise_cost_more(self):
        two = tools.score_acoustics(["crowded"])["score"]
        one = tools.score_acoustics(["crowded"], single_sided=True)["score"]
        self.assertGreater(one, two)


class DirectoryTests(unittest.TestCase):
    def test_keyword_match(self):
        names = [o["name"] for o in tools.match_known_orgs("my yale pharmacy calls every Monday")]
        self.assertEqual(names, ["Yale Health Pharmacy (55 Lock Street, New Haven)"])
        self.assertEqual(tools.match_known_orgs("my local bank"), [])

    def test_every_entry_is_sourced_and_dated(self):
        for o in tools.KNOWN_ORGS:
            self.assertTrue(o["sources"] and all(u.startswith("https://") for u in o["sources"]), o["id"])
            self.assertRegex(o["checked"], r"^\d{4}-\d{2}-\d{2}$")
            for e in o["emails"]:
                self.assertRegex(e["address"], r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$")

    def test_every_org_type_has_where_to_look(self):
        self.assertEqual(set(tools.WHERE_TO_LOOK), set(tools.ORG_TYPES))


class CircleTests(unittest.TestCase):
    PEOPLE = ["Mom - soft voice", "Dad: talks most", "Raj (fast talker)", "Priya", "Uncle - loud"]

    def seats(self, side, shape="round"):
        r = tools.plan_seating(shape, tools.parse_people(self.PEOPLE), side, ["tv"])
        return {s["name"]: s["seat"] for s in r["seats"]}, r

    def test_parse_tags(self):
        tags = {p["name"]: p["tag"] for p in tools.parse_people(self.PEOPLE)}
        self.assertEqual(tags, {"Mom": "quiet", "Dad": "most", "Raj": "hard", "Priya": "", "Uncle": "loud"})

    def test_left_side_hearing(self):
        seat, r = self.seats("left")
        self.assertEqual(seat["You"], 0)
        self.assertEqual(seat["Dad"], 1)          # most wanted: right next to you, on your left
        self.assertEqual(seat["Mom"], 3)          # soft voice: directly across, face visible
        self.assertEqual(seat["Uncle"], 5)        # loud: your weaker side is fine
        self.assertLess(r["seats"][1]["x"], 0.5)  # seat 1 is drawn on your left
        self.assertIn("back to the noise (TV)", " ".join(r["tips"]))

    def test_right_side_mirrors(self):
        seat, _ = self.seats("right")
        self.assertEqual(seat["Dad"], 5)

    def test_both_ears_puts_faces_across(self):
        seat, _ = self.seats(None)
        self.assertEqual(seat["Dad"], 3)

    def test_long_table_and_limits(self):
        seat, r = self.seats("left", "long")
        self.assertEqual(seat["Dad"], 1)
        self.assertEqual(len(r["seats"]), 6)
        many = tools.plan_seating("round", tools.parse_people([f"P{i}" for i in range(20)]))
        self.assertEqual(many["people"], tools.MAX_PEOPLE)


class ScoreMentionTests(unittest.TestCase):
    """Found in a live run: the model wrote the pre-guard score into its text, which went stale."""

    def test_stale_score_text_is_removed(self):
        from hearing.agents import SCORE_MENTION
        text = "Ask for a back booth. Score: 5.5/10. Verdict: Manageable with a good seat."
        self.assertEqual(SCORE_MENTION.sub("", text).strip(), "Ask for a back booth.")
        self.assertNotIn("3.5", SCORE_MENTION.sub("", "Photo is unclear; tool score: 3.5, verdict: Easy."))
        self.assertEqual(SCORE_MENTION.sub("", "Sit facing the wall."), "Sit facing the wall.")


if __name__ == "__main__":
    unittest.main()
