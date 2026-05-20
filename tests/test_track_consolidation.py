from accident_vlm.modules.track_consolidation import consolidate_tracks


def test_consolidate_tracks_merges_same_type_close_fragments() -> None:
    tracks = [
        {
            "track_id": "T1",
            "type": "승용차",
            "positions": [
                {"frame_id": "frame_000000", "bbox": [0, 0, 20, 20]},
                {"frame_id": "frame_000030", "bbox": [20, 0, 40, 20]},
            ],
        },
        {
            "track_id": "T9",
            "type": "승용차",
            "positions": [
                {"frame_id": "frame_000045", "bbox": [35, 0, 55, 20]},
            ],
        },
    ]

    consolidated = consolidate_tracks(tracks)

    assert len(consolidated) == 1
    assert consolidated[0]["consolidated_from"] == ["T1", "T9"]
    assert len(consolidated[0]["positions"]) == 3
    assert consolidated[0]["tracking_method"] == "consolidated_track"


def test_consolidate_tracks_keeps_distant_fragments_separate() -> None:
    consolidated = consolidate_tracks(
        [
            {
                "track_id": "T1",
                "type": "보행자",
                "positions": [{"frame_id": "frame_000000", "bbox": [0, 0, 10, 10]}],
            },
            {
                "track_id": "T2",
                "type": "보행자",
                "positions": [{"frame_id": "frame_000010", "bbox": [500, 0, 520, 20]}],
            },
        ]
    )

    assert len(consolidated) == 2
