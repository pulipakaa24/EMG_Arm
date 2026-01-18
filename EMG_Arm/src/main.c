
#include <freertos/FreeRTOS.h>
#include "driver/gpio.h"
#include "driver/ledc.h"

// Finger servo pin mappings
#define thumbServoPin   GPIO_NUM_1
#define indexServoPin   GPIO_NUM_4
#define middleServoPin  GPIO_NUM_5
#define ringServoPin    GPIO_NUM_6
#define pinkyServoPin   GPIO_NUM_7

// LEDC channels (one per servo)
#define thumbChannel    LEDC_CHANNEL_0
#define indexChannel    LEDC_CHANNEL_1
#define middleChannel   LEDC_CHANNEL_2
#define ringChannel     LEDC_CHANNEL_3
#define pinkyChannel    LEDC_CHANNEL_4
#define deg180 2048
#define deg0 430

// Arrays for cleaner initialization
const int servoPins[] = {thumbServoPin, indexServoPin, middleServoPin, ringServoPin, pinkyServoPin};
const int servoChannels[] = {thumbChannel, indexChannel, middleChannel, ringChannel, pinkyChannel};
const int numServos = 5;

void servoInit() {
  // LEDC timer configuration (shared by all servos)
  ledc_timer_config_t ledc_timer = {};
  ledc_timer.speed_mode = LEDC_LOW_SPEED_MODE;
  ledc_timer.timer_num = LEDC_TIMER_0;
  ledc_timer.duty_resolution = LEDC_TIMER_14_BIT;
  ledc_timer.freq_hz = 50;
  ledc_timer.clk_cfg = LEDC_AUTO_CLK;
  ESP_ERROR_CHECK(ledc_timer_config(&ledc_timer));

  // Initialize each finger servo channel
  for (int i = 0; i < numServos; i++) {
    ledc_channel_config_t ledc_channel = {};
    ledc_channel.speed_mode = LEDC_LOW_SPEED_MODE;
    ledc_channel.channel = servoChannels[i];
    ledc_channel.timer_sel = LEDC_TIMER_0;
    ledc_channel.intr_type = LEDC_INTR_DISABLE;
    ledc_channel.gpio_num = servoPins[i];
    ledc_channel.duty = deg0; // Start with fingers open
    ledc_channel.hpoint = 0;
    ESP_ERROR_CHECK(ledc_channel_config(&ledc_channel));
  }
}

// Flex functions (move to 180 degrees - finger closed)
void flexThumb() {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, thumbChannel, deg180);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, thumbChannel);
}

void flexIndex() {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, indexChannel, deg180);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, indexChannel);
}

void flexMiddle() {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, middleChannel, deg180);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, middleChannel);
}

void flexRing() {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, ringChannel, deg180);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, ringChannel);
}

void flexPinky() {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, pinkyChannel, deg180);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, pinkyChannel);
}

// Unflex functions (move to 0 degrees - finger open)
void unflexThumb() {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, thumbChannel, deg0);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, thumbChannel);
}

void unflexIndex() {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, indexChannel, deg0);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, indexChannel);
}

void unflexMiddle() {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, middleChannel, deg0);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, middleChannel);
}

void unflexRing() {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, ringChannel, deg0);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, ringChannel);
}

void unflexPinky() {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, pinkyChannel, deg0);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, pinkyChannel);
}

// Combo functions
void makeFist() {
  // Set all duties first
  ledc_set_duty(LEDC_LOW_SPEED_MODE, thumbChannel, deg180);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, indexChannel, deg180);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, middleChannel, deg180);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, ringChannel, deg180);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, pinkyChannel, deg180);
  // Update all at once
  ledc_update_duty(LEDC_LOW_SPEED_MODE, thumbChannel);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, indexChannel);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, middleChannel);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, ringChannel);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, pinkyChannel);
}

void openHand() {
  // Set all duties first
  ledc_set_duty(LEDC_LOW_SPEED_MODE, thumbChannel, deg0);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, indexChannel, deg0);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, middleChannel, deg0);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, ringChannel, deg0);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, pinkyChannel, deg0);
  // Update all at once
  ledc_update_duty(LEDC_LOW_SPEED_MODE, thumbChannel);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, indexChannel);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, middleChannel);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, ringChannel);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, pinkyChannel);
}

void individualFingerDemo(int delay_ms){
  flexThumb();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    unflexThumb();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));

    flexIndex();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    unflexIndex();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));

    flexMiddle();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    unflexMiddle();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));

    flexRing();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    unflexRing();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));

    flexPinky();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    unflexPinky();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
}

void closeOpenDemo(int delay_ms){
    makeFist();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    openHand();
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
}

void app_main() {
  servoInit();

  // Demo: flex and unflex each finger in sequence
  // while(1) {
  //   individualFingerDemo(1000);
  // }

  // Demo: close and open hand
  while(1) {
    closeOpenDemo(1000);
  }
}