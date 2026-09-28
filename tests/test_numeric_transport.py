"""Bounded numeric frames preserve values without Python deserialization."""

import struct
import unittest

import numpy as np

from examples.elitelist_papers.poker.codec import (
    MAX_ARRAY_BYTES,
    MAX_MESSAGE,
    frame_size,
    pack,
    unpack,
)


class NumericTransportTests(unittest.TestCase):
    def test_roundtrip_numeric_and_generic_messages(self):
        values = [
            np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2],
            np.array([0.5], dtype=">f8"),
            np.array(2.0),
            np.empty((0, 2), dtype=np.float64),
            3,
            np.int64(4),
            {"nested": (np.array([1, 2]), "text")},
        ]
        for value in values:
            for key in ("observation", "action"):
                message = {"command": "act", key: value} if key == "observation" else {key: value}
                packet = pack(message)
                self.assertEqual(frame_size(packet[:4]), len(packet) - 4)
                result = unpack(packet[4:])[key]
                if isinstance(value, dict):
                    np.testing.assert_array_equal(result["nested"][0], value["nested"][0])
                    self.assertEqual(result["nested"][1], "text")
                else:
                    np.testing.assert_array_equal(result, value)
                    if isinstance(value, np.ndarray):
                        self.assertEqual(result.dtype, value.dtype)
                        self.assertEqual(result.shape, value.shape)
        for message in ({"command": "close"}, {"ok": True}, {"error": "bad action"}):
            self.assertEqual(unpack(pack(message)[4:]), message)

    def test_rejects_invalid_lengths_dtypes_shapes_and_payloads(self):
        for header in (b"", b"abc", struct.pack("!I", 0), struct.pack("!I", MAX_MESSAGE + 1)):
            with self.subTest(header=header), self.assertRaises(ValueError):
                frame_size(header)
        for body in (
            b"",
            b"X",
            b'A["O",[1]]\n12345678',
            b'A["<i4",[-1]]\n',
            b'A["<i4",[1]]\n123',
            b'A["<i4",[1]]\n12345',
            b'A["<i4",[true]]\n1234',
            b'A["<i4",[99999999]]\n',
            b"A" + b" " * 1025 + b"\n",
            b"J[]\n",
        ):
            with self.subTest(body=body[:60]), self.assertRaises(ValueError):
                unpack(body)
        for value in (np.array([object()]), np.zeros(MAX_ARRAY_BYTES + 1, dtype=np.uint8)):
            with self.assertRaises(ValueError):
                pack({"action": value})
