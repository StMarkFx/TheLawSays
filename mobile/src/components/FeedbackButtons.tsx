import { Feather } from "@expo/vector-icons";
import React from "react";
import { StyleSheet, TouchableOpacity, View } from "react-native";

import { colors } from "../theme";

interface FeedbackButtonsProps {
  onFeedback: (rating: "thumbs_up" | "thumbs_down") => void;
  disabled?: boolean;
}

export function FeedbackButtons({ onFeedback, disabled }: FeedbackButtonsProps) {
  return (
    <View style={styles.container}>
      <TouchableOpacity
        onPress={() => onFeedback("thumbs_up")}
        style={[styles.button, styles.buttonSpacing, disabled && styles.buttonDisabled]}
        disabled={disabled}
        accessibilityLabel="Thumbs up"
      >
        <Feather name="thumbs-up" size={16} color={colors.muted} />
      </TouchableOpacity>
      <TouchableOpacity
        onPress={() => onFeedback("thumbs_down")}
        style={[styles.button, disabled && styles.buttonDisabled]}
        disabled={disabled}
        accessibilityLabel="Thumbs down"
      >
        <Feather name="thumbs-down" size={16} color={colors.muted} />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row"
  },
  button: {
    height: 36,
    width: 36,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: "center",
    justifyContent: "center"
  },
  buttonSpacing: {
    marginRight: 12
  },
  buttonDisabled: {
    opacity: 0.4
  }
});
