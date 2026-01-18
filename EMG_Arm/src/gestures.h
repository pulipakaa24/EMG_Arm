/**
 * @file gestures.h
 * @brief Gesture control interface for the EMG-controlled robotic hand.
 *
 * This module provides high-level gesture functions for controlling the
 * robotic hand. It builds upon the low-level servo control module to
 * implement intuitive finger movements and hand gestures.
 *
 * Gesture Categories:
 *   - Individual finger control (flex/unflex each finger)
 *   - Composite gestures (fist, open hand, etc.)
 *   - Demo/test sequences
 */

#ifndef GESTURES_H
#define GESTURES_H

#include <stdint.h>

/*******************************************************************************
 * Individual Finger Control - Flex Functions
 *
 * Flex functions move a finger to the closed (180 degree) position.
 ******************************************************************************/

/**
 * @brief Flex the thumb (move to closed position).
 */
void flex_thumb(void);

/**
 * @brief Flex the index finger (move to closed position).
 */
void flex_index(void);

/**
 * @brief Flex the middle finger (move to closed position).
 */
void flex_middle(void);

/**
 * @brief Flex the ring finger (move to closed position).
 */
void flex_ring(void);

/**
 * @brief Flex the pinky finger (move to closed position).
 */
void flex_pinky(void);

/*******************************************************************************
 * Individual Finger Control - Unflex Functions
 *
 * Unflex functions move a finger to the extended (0 degree) position.
 ******************************************************************************/

/**
 * @brief Unflex the thumb (move to extended position).
 */
void unflex_thumb(void);

/**
 * @brief Unflex the index finger (move to extended position).
 */
void unflex_index(void);

/**
 * @brief Unflex the middle finger (move to extended position).
 */
void unflex_middle(void);

/**
 * @brief Unflex the ring finger (move to extended position).
 */
void unflex_ring(void);

/**
 * @brief Unflex the pinky finger (move to extended position).
 */
void unflex_pinky(void);

/*******************************************************************************
 * Composite Gestures
 *
 * These functions control multiple fingers simultaneously to form gestures.
 ******************************************************************************/

/**
 * @brief Close all fingers to form a fist.
 *
 * All five fingers move to the flexed position simultaneously.
 */
void gesture_make_fist(void);

/**
 * @brief Open the hand fully.
 *
 * All five fingers move to the extended position simultaneously.
 */
void gesture_open_hand(void);

/*******************************************************************************
 * Demo/Test Sequences
 *
 * These functions provide demonstration sequences for testing servo operation.
 ******************************************************************************/

/**
 * @brief Demo sequence: flex and unflex each finger individually.
 *
 * Cycles through each finger, flexing and unflexing with a delay
 * between each movement. Useful for testing individual servo operation.
 *
 * @param delay_ms Delay in milliseconds between each movement.
 */
void demo_individual_fingers(uint32_t delay_ms);

/**
 * @brief Demo sequence: repeatedly close and open the hand.
 *
 * Alternates between making a fist and opening the hand.
 * Useful for testing simultaneous servo operation.
 *
 * @param delay_ms Delay in milliseconds between fist and open positions.
 */
void demo_close_open(uint32_t delay_ms);

#endif /* GESTURES_H */
