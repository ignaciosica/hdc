import unittest
from hdc import hdv, bind, bundle, bundleS, cosine_similarity

DIMENSIONS = 10000
EQUALITY_THRESHOLD = 0.9
SIMILARITY_THRESHOLD = 0.1
ORTOGONALITY_THRESHOLD = 0.05


class TestHdv(unittest.TestCase):
    def test_hdv(self):
        H = hdv(DIMENSIONS)

        self.assertEqual(len(H), DIMENSIONS)

    def test_cosine_similarity_self(self):
        A = hdv(DIMENSIONS)

        self.assertGreater(abs(cosine_similarity(A, A)), EQUALITY_THRESHOLD)

    def test_cosine_similarity_orthogonal(self):
        A, B = hdv(DIMENSIONS), hdv(DIMENSIONS)

        self.assertLess(abs(cosine_similarity(A, B)), ORTOGONALITY_THRESHOLD)


class TestBind(unittest.TestCase):
    def test_bind(self):
        A, B = hdv(DIMENSIONS), hdv(DIMENSIONS)
        H = bind([A, B])

        self.assertEqual(len(H), DIMENSIONS)

    def test_bind_xs(self):
        A, B, C, D = hdv(DIMENSIONS), hdv(DIMENSIONS), hdv(DIMENSIONS), hdv(DIMENSIONS)
        H = bind([A, B, C, D])

        self.assertEqual(len(H), DIMENSIONS)

    def test_cosine_similarity_bind(self):
        A, B = hdv(DIMENSIONS), hdv(DIMENSIONS)

        H = bind([A, B])

        self.assertLess(abs(cosine_similarity(A, H)), ORTOGONALITY_THRESHOLD)
        self.assertLess(abs(cosine_similarity(B, H)), ORTOGONALITY_THRESHOLD)


class TestBundle(unittest.TestCase):
    def test_bundle(self):
        A, B = hdv(DIMENSIONS), hdv(DIMENSIONS)
        H = bind([A, B])

        self.assertEqual(len(H), DIMENSIONS)

    def test_bundle_xs(self):
        A, B, C, D = hdv(DIMENSIONS), hdv(DIMENSIONS), hdv(DIMENSIONS), hdv(DIMENSIONS)

        H = bind([A, B, C, D])

        self.assertEqual(len(H), DIMENSIONS)

    def test_cosine_similarity_bundle(self):
        A, B = hdv(DIMENSIONS), hdv(DIMENSIONS)

        H = bundle([A, B])

        self.assertGreater(abs(cosine_similarity(A, H)), SIMILARITY_THRESHOLD)
        self.assertGreater(abs(cosine_similarity(B, H)), SIMILARITY_THRESHOLD)

    def test_cosine_similarity_bundle_xs(self):
        xs = [hdv(DIMENSIONS) for _ in range(250)]

        H = bundle(xs)

        for x in xs:
            self.assertGreater(
                cosine_similarity(x, H),
                cosine_similarity(x, hdv(DIMENSIONS)),
            )


class TestBasicArithmetic(unittest.TestCase):
    def test_distributivity(self):
        A, B, C = hdv(DIMENSIONS), hdv(DIMENSIONS), hdv(DIMENSIONS)

        H = bind([A, bundle([B, C])])
        D = bundle([bind([A, B]), bind([A, C])])

        self.assertGreater(cosine_similarity(H, D), EQUALITY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
