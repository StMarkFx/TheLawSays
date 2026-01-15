import { Feather } from "@expo/vector-icons";
import React, { useState } from "react";
import { StyleSheet, TextInput, TouchableOpacity, View } from "react-native";

import { colors, fonts } from "../theme";

interface ComposerProps {
  onSubmit: (message: string) => void;
  disabled?: boolean;
}

export function Composer({ onSubmit, disabled }: ComposerProps) {
  const [value, setValue] = useState("");

  const canSend = value.trim().length > 0 && !disabled;

  const handleSend = () => {
    if (!canSend) return;
    onSubmit(value.trim());
    setValue("");
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Ask TheLaw"
        placeholderTextColor="#6d6d6d"
        value={value}
        onChangeText={setValue}
        editable={!disabled}
        multiline
      />
      <TouchableOpacity
        onPress={handleSend}
        style={[styles.sendButton, canSend ? styles.sendButtonActive : styles.sendButtonDisabled]}
        disabled={!canSend}
        accessibilityLabel="Send message"
      >
        <Feather name="send" size={18} color={canSend ? colors.accent : colors.muted} />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    alignItems: "flex-end",
    borderRadius: 22,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.input,
    paddingHorizontal: 16,
    paddingVertical: 10
  },
  input: {
    flex: 1,
    minHeight: 36,
    maxHeight: 120,
    marginRight: 12,
    color: colors.foreground,
    fontSize: 16,
    fontFamily: fonts.regular
  },
  sendButton: {
    height: 44,
    width: 44,
    borderRadius: 22,
    borderWidth: 1,
    alignItems: "center",
    justifyContent: "center"
  },
  sendButtonActive: {
    borderColor: colors.accent,
    backgroundColor: "rgba(0,229,204,0.08)"
  },
  sendButtonDisabled: {
    borderColor: colors.border
  }
});
